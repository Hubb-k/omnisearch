use core_lib::crypto;
use core_lib::index::hnsw::HnswIndex;
use core_lib::vectorize::minilm::MiniLM;
use std::fs;

const DATA_DIR: &str = "data";
const CHUNK_SIZE: usize = 500;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Использование: importer <папка>");
        return Ok(());
    }

    // Инициализируем обфускацию
    crypto::init();

    let folder = &args[1];
    let mut model = MiniLM::load()?;
    let mut index = HnswIndex::new(DATA_DIR)?;

    println!("[Импортер] Индекс загружен: {} записей", index.len());

    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(
            ext,
            "txt" | "md" | "rs" | "toml" | "json" | "js" | "ts" | "py"
        ) {
            continue;
        }

        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        println!("[Файл] {}", filename);

        let content = fs::read(&path).unwrap_or_default();
        let text = if ext == "txt" {
            if let Ok(s) = String::from_utf8(content.clone()) {
                s
            } else {
                encoding_rs::WINDOWS_1251.decode(&content).0.into_owned()
            }
        } else {
            String::from_utf8(content).unwrap_or_default()
        };

        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        let mut file_count = 0;

        while i < chars.len() {
            let end = (i + CHUNK_SIZE).min(chars.len());
            let chunk: String = chars[i..end].iter().collect();
            let chunk = chunk.trim().to_string();

            if chunk.len() > 50 {
                match model.embed(&chunk) {
                    Ok(vec) => {
                        // Обфускация перед записью в индекс
                        let vec = crypto::permute(&vec);
                        let _ = index.add(&chunk, &filename, &vec);
                        file_count += 1;
                        if file_count % 200 == 0 {
                            println!("  → {} чанков", file_count);
                            let _ = index.save(DATA_DIR);
                        }
                    }
                    Err(e) => eprintln!("[Ошибка] {}", e),
                }
            }
            i += CHUNK_SIZE - 100;
        }

        println!("  → Готово: {} чанков", file_count);
        index.save(DATA_DIR)?;
    }

    println!("[Готово] Всего в индексе: {} записей", index.len());
    Ok(())
}
