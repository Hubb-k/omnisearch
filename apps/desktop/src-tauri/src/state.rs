use core_lib::vectorize::minilm::MiniLM;
use core_lib::index::hnsw::{HnswIndex, SearchResult};

const DATA_DIR: &str = "../../data";

pub struct AppState {
    model: MiniLM,
    index: HnswIndex,
}

impl AppState {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let model = MiniLM::load()?;
        let index = HnswIndex::new(DATA_DIR)?;
        Ok(Self { model, index })
    }

    pub fn search(&mut self, query: &str) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let vec = self.model.embed(query)?;
        self.index.search(&vec, 10)
    }
}