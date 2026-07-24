#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use std::sync::{Arc, Mutex};

mod state;
use state::AppState;

#[tauri::command]
fn search(query: String, state: tauri::State<Arc<Mutex<AppState>>>) -> Vec<serde_json::Value> {
    let mut state = state.lock().unwrap();
    match state.search(&query) {
        Ok(results) => results.into_iter().map(|r| serde_json::json!({
            "text": r.text,
            "source": r.source,
            "distance": r.distance,
            "timestamp": r.timestamp,
        })).collect(),
        Err(e) => {
            eprintln!("[Ошибка поиска] {}", e);
            vec![]
        }
    }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let state = Arc::new(Mutex::new(AppState::new().expect("Не удалось инициализировать состояние")));
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![search])
        .run(tauri::generate_context!())
        .expect("Ошибка запуска Tauri");
}