use core_lib::index::hnsw::HnswIndex;
use core_lib::vectorize::minilm::MiniLM;
use std::sync::{Arc, Mutex};
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

struct AppState {
    model: MiniLM,
    index: HnswIndex,
    data_dir: String,
}

#[tauri::command]
fn search(query: String, state: tauri::State<Arc<Mutex<AppState>>>) -> Vec<serde_json::Value> {
    let mut state = state.lock().unwrap();

    let vec = match state.model.embed(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[Ошибка] {}", e);
            return vec![];
        }
    };
    let vec = core_lib::crypto::permute(&vec);

    let vector_hits = state.index.search(&vec, 10).unwrap_or_default();

    let fts_query = query
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let fts_hits = state.index.fts_search(&fts_query, 10).unwrap_or_default();

    let mut seen = std::collections::HashSet::new();
    let mut combined: Vec<serde_json::Value> = Vec::new();

    for r in vector_hits.iter() {
        if seen.insert(r.id) {
            combined.push(serde_json::json!({
                "id": r.id,
                "text": r.text,
                "source": r.source,
                "distance": r.distance,
                "timestamp": r.timestamp,
                "match_type": "vector",
            }));
        }
    }
    for r in fts_hits.iter() {
        if seen.insert(r.id) {
            combined.push(serde_json::json!({
                "id": r.id,
                "text": r.text,
                "source": r.source,
                "distance": r.distance,
                "timestamp": r.timestamp,
                "match_type": "fts",
            }));
        }
    }

    let high_quality: Vec<_> = vector_hits
        .iter()
        .filter(|r| (1.0 - r.distance) >= 0.78)
        .collect();

    if !high_quality.is_empty() {
        let chosen_ids: Vec<u64> = high_quality.iter().map(|r| r.id).collect();
        let rejected_ids: Vec<u64> = vector_hits
            .iter()
            .filter(|r| (1.0 - r.distance) < 0.78)
            .map(|r| r.id)
            .collect();

        for &chosen_id in &chosen_ids {
            if let Err(e) = state.index.add_feedback(&query, chosen_id, &rejected_ids) {
                eprintln!("[AutoFeedback] Ошибка: {}", e);
            }
        }
        let count = state.index.feedback_count();
        eprintln!(
            "[AutoFeedback] chosen={} rejected={} пар={}",
            chosen_ids.len(),
            rejected_ids.len(),
            count
        );
        if core_lib::training::should_train(count) {
            let data_dir = state.data_dir.clone();
            let model_dir = format!("{}/../../../models", env!("CARGO_MANIFEST_DIR"));
            core_lib::training::spawn_finetune(&data_dir, &model_dir);
        }
    }

    eprintln!(
        "[Search] vector={} fts={} total={}",
        vector_hits.len(),
        fts_hits.len(),
        combined.len()
    );
    combined
}

#[tauri::command]
fn feedback(
    query: String,
    chosen_id: u64,
    rejected_ids: Vec<u64>,
    state: tauri::State<Arc<Mutex<AppState>>>,
) -> bool {
    let s = state.lock().unwrap();
    if let Err(e) = s.index.add_feedback(&query, chosen_id, &rejected_ids) {
        eprintln!("[Feedback] Ошибка: {}", e);
        return false;
    }
    let count = s.index.feedback_count();
    eprintln!("[Feedback] Сохранено. Всего пар: {}", count);
    if core_lib::training::should_train(count) {
        let data_dir = s.data_dir.clone();
        let model_dir = format!("{}/../../../models", env!("CARGO_MANIFEST_DIR"));
        drop(s);
        core_lib::training::spawn_finetune(&data_dir, &model_dir);
    }
    true
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let root = env!("CARGO_MANIFEST_DIR");
    let data_dir = format!("{}/../../../data", root);
    let model_path = format!("{}/../../../models/model.onnx", root);
    let tokenizer_path = format!("{}/../../../models/tokenizer.json", root);

    std::env::set_var("ORT_MODEL_PATH", &model_path);
    std::env::set_var("ORT_TOKENIZER_PATH", &tokenizer_path);

    core_lib::crypto::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(move |app| {
            let state = Arc::new(Mutex::new(AppState {
                model: MiniLM::load().expect("MiniLM load failed"),
                index: HnswIndex::new(&data_dir).expect("HnswIndex init failed"),
                data_dir: data_dir.clone(),
            }));
            app.manage(state.clone());

            let shortcut = Shortcut::new(Some(Modifiers::ALT), Code::Space);
            let app_handle = app.handle().clone();
            app.global_shortcut()
                .on_shortcut(shortcut, move |_app, _shortcut, event| {
                    if event.state == tauri_plugin_global_shortcut::ShortcutState::Pressed {
                        if let Some(window) = app_handle.get_webview_window("main") {
                            if window.is_visible().unwrap_or(false) {
                                let _ = window.hide();
                            } else {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                    }
                })?;

            let state_ws = state.clone();
            core_lib::capture::websocket::start_ws_server(move |text: String, source: String| {
                if text.trim().len() < 10 {
                    return;
                }
                let mut s = state_ws.lock().unwrap();
                let vec = match s.model.embed(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[WS Ошибка] {}", e);
                        return;
                    }
                };
                let vec = core_lib::crypto::permute(&vec);
                if let Ok(results) = s.index.search(&vec, 1) {
                    if let Some(r) = results.first() {
                        if r.distance < 0.05 {
                            return;
                        }
                    }
                }
                match s.index.add(&text, &source, &vec) {
                    Ok(id) => println!("[Browser] id={} source={}", id, source),
                    Err(e) => eprintln!("[WS Индекс] {}", e),
                }
            });

            let state_cb = state.clone();
            std::thread::spawn(move || {
                if let Err(e) = core_lib::capture::clipboard::start_listener(move |text: String| {
                    if text.trim().len() < 10 {
                        return;
                    }
                    let mut s = state_cb.lock().unwrap();
                    let vec = match s.model.embed(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("[Ошибка векторизации] {}", e);
                            return;
                        }
                    };
                    let vec = core_lib::crypto::permute(&vec);
                    if let Ok(results) = s.index.search(&vec, 1) {
                        if let Some(r) = results.first() {
                            if r.distance < 0.05 {
                                return;
                            }
                        }
                    }
                    let data_dir = s.data_dir.clone();
                    match s.index.add(&text, "clipboard", &vec) {
                        Ok(id) => {
                            println!("[Захват] id={} chars={}", id, text.chars().count());
                            let _ = s.index.save(&data_dir);
                        }
                        Err(e) => eprintln!("[Ошибка индекса] {}", e),
                    }
                }) {
                    eprintln!("[Синапс] Ошибка: {:?}", e);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![search, feedback])
        .build(tauri::generate_context!())
        .expect("Tauri error")
        .run(move |app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app_handle.try_state::<Arc<Mutex<AppState>>>() {
                    let s = state.lock().unwrap();
                    if let Err(e) = s.index.save(&s.data_dir) {
                        eprintln!("[Shutdown] Ошибка сохранения: {}", e);
                    } else {
                        println!("[Shutdown] Индекс сохранён.");
                    }
                }
            }
        });
}
