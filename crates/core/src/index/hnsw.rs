use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use rusqlite::{Connection, params};
use chrono::Utc;
use crate::crypto;

pub struct HnswIndex {
    index: Index,
    db: Connection,
    next_id: u64,
}

impl HnswIndex {
    pub fn new(data_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(data_dir)?;

        let options = IndexOptions {
            dimensions: 384,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            ..Default::default()
        };

        let index_path = format!("{}/vectors.usearch", data_dir);
        let db_path = format!("{}/meta.sqlite", data_dir);

        let index = Index::new(&options)?;

        if std::path::Path::new(&index_path).exists() {
            index.load(&index_path)?;
            index.reserve(index.size() + 100_000)?;
        } else {
            index.reserve(100_000)?;
        }

        let db = Connection::open(&db_path)?;

        let key = format!("{:016x}", crypto::get_seed());
        db.pragma_update(None, "key", &key)?;

        db.execute_batch("
            CREATE TABLE IF NOT EXISTS entries (
                id        INTEGER PRIMARY KEY,
                text      TEXT NOT NULL,
                source    TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                text,
                content='entries',
                content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
                INSERT INTO entries_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TABLE IF NOT EXISTS feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query        TEXT NOT NULL,
                chosen_id    INTEGER NOT NULL,
                rejected_ids TEXT NOT NULL,
                timestamp    TEXT NOT NULL
            );
        ")?;

        let next_id: u64 = db.query_row(
            "SELECT COALESCE(MAX(id) + 1, 0) FROM entries",
            [],
            |row| row.get(0),
        )?;

        Ok(Self { index, db, next_id })
    }

    pub fn add(&mut self, text: &str, source: &str, vector: &[f32]) -> Result<u64, Box<dyn std::error::Error>> {
        let id = self.next_id;
        self.next_id += 1;
        self.index.add(id, vector)?;
        self.db.execute(
            "INSERT INTO entries (id, text, source, timestamp) VALUES (?1, ?2, ?3, ?4)",
            params![id, text, source, Utc::now().to_rfc3339()],
        )?;
        Ok(id)
    }

    pub fn search(&self, vector: &[f32], top_k: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let results = self.index.search(vector, top_k)?;
        let mut out = Vec::new();
        for (&id, &dist) in results.keys.iter().zip(results.distances.iter()) {
            let (text, source, timestamp) = self.db.query_row(
                "SELECT text, source, timestamp FROM entries WHERE id = ?1",
                params![id],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                )),
            )?;
            out.push(SearchResult { id, distance: dist, text, source, timestamp });
        }
        Ok(out)
    }

    pub fn fts_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let mut stmt = self.db.prepare(
            "SELECT e.id, e.text, e.source, e.timestamp, f.rank
             FROM entries_fts f
             JOIN entries e ON e.id = f.rowid
             WHERE entries_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2"
        )?;
        let results = stmt.query_map(
            params![query, top_k as i64],
            |row| Ok((
                row.get::<_, u64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, f64>(4)?,
            )),
        )?
        .filter_map(|r| r.ok())
        .map(|(id, text, source, timestamp, rank)| {
            let distance = (rank.abs() / 20.0).min(1.0) as f32;
            SearchResult { id, text, source, timestamp, distance }
        })
        .collect();
        Ok(results)
    }

    pub fn add_feedback(&self, query: &str, chosen_id: u64, rejected_ids: &[u64]) -> Result<(), Box<dyn std::error::Error>> {
        let rejected = serde_json::to_string(rejected_ids)?;
        self.db.execute(
            "INSERT INTO feedback (query, chosen_id, rejected_ids, timestamp) VALUES (?1, ?2, ?3, ?4)",
            params![query, chosen_id as i64, rejected, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn feedback_count(&self) -> usize {
        self.db.query_row("SELECT COUNT(*) FROM feedback", [], |r| r.get(0)).unwrap_or(0)
    }

    pub fn save(&self, data_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
        let index_path = format!("{}/vectors.usearch", data_dir);
        self.index.save(&index_path)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.index.size()
    }
}

pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
    pub text: String,
    pub source: String,
    pub timestamp: String,
}