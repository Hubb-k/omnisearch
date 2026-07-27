use core_lib::index::hnsw::HnswIndex;
use core_lib::vectorize::minilm::MiniLM;
use std::sync::{Arc, Mutex};
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

struct AppStateInner {
    model: MiniLM,
    index: HnswIndex,
    data_dir: String,
}

struct AppState {
    inner: Option<AppStateInner>,
}

#[tauri::command]
fn unlock(
    password: String,
    state: tauri::State<Arc<Mutex<AppState>>>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    let root = env!("CARGO_MANIFEST_DIR");
    let data_dir = format!("{}/../../../data", root);
    let model_dir = format!("{}/../../../models", root);

    core_lib::crypto::init_with_password(&data_dir, &password)?;
    core_lib::adapter::init(&model_dir);

    let model = MiniLM::load().map_err(|e| format!("MiniLM load failed: {}", e))?;
    let index = HnswIndex::new(&data_dir).map_err(|e| format!("HnswIndex init failed: {}", e))?;

    {
        let mut s = state.lock().unwrap();
        s.inner = Some(AppStateInner { model, index, data_dir });
    }

    if let Some(pw_window) = app.get_webview_window("password") {
        let _ = pw_window.close();
    }
    if let Some(main_window) = app.get_webview_window("main") {
        let _ = main_window.show();
        let _ = main_window.set_focus();
    }

    Ok(())
}

#[tauri::command]
fn search(
    query: String,
    state: tauri::State<Arc<Mutex<AppState>>>,
) -> Vec<serde_json::Value> {
    let mut state = state.lock().unwrap();
    let inner = match state.inner.as_mut() {
        Some(s) => s,
        None => return vec![],
    };

    let vec = match inner.model.embed(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[Ошибка] {}", e);
            return vec![];
        }
    };
    let vec = core_lib::adapter::apply_if_loaded(&vec);
    let vec = core_lib::crypto::permute(&vec);

    let vector_hits = inner.index.search(&vec, 10).unwrap_or_default();

    let fts_query = query
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let fts_hits = inner.index.fts_search(&fts_query, 10).unwrap_or_default();

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
            if let Err(e) = inner.index.add_feedback(&query, chosen_id, &rejected_ids) {
                eprintln!("[AutoFeedback] Ошибка: {}", e);
            }
        }
        let count = inner.index.feedback_count();
        eprintln!(
            "[AutoFeedback] chosen={} rejected={} пар={}",
            chosen_ids.len(),
            rejected_ids.len(),
            count
        );
        if core_lib::training::should_train(count) {
            let model_dir = format!("{}/../../../models", env!("CARGO_MANIFEST_DIR"));
            match inner.index.get_triplets(200) {
                Ok(triplets) => core_lib::training::spawn_adapter_training(triplets, model_dir),
                Err(e) => eprintln!("[Train] Ошибка получения триплетов: {}", e),
            }
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
    let mut s = state.lock().unwrap();
    let inner = match s.inner.as_mut() {
        Some(s) => s,
        None => return false,
    };

    if let Err(e) = inner.index.add_feedback(&query, chosen_id, &rejected_ids) {
        eprintln!("[Feedback] Ошибка: {}", e);
        return false;
    }
    let count = inner.index.feedback_count();
    eprintln!("[Feedback] Сохранено. Всего пар: {}", count);

    if core_lib::training::should_train(count) {
        let model_dir = format!("{}/../../../models", env!("CARGO_MANIFEST_DIR"));
        match inner.index.get_triplets(200) {
            Ok(triplets) => {
                drop(s);
                core_lib::training::spawn_adapter_training(triplets, model_dir);
            }
            Err(e) => eprintln!("[Train] Ошибка получения триплетов: {}", e),
        }
    }
    true
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let root = env!("CARGO_MANIFEST_DIR");
    let model_path = format!("{}/../../../models/model.onnx", root);
    let tokenizer_path = format!("{}/../../../models/tokenizer.json", root);

    std::env::set_var("ORT_MODEL_PATH", &model_path);
    std::env::set_var("ORT_TOKENIZER_PATH", &tokenizer_path);

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(move |app| {
            let state = Arc::new(Mutex::new(AppState { inner: None }));
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
                let inner = match s.inner.as_mut() {
                    Some(i) => i,
                    None => return,
                };
                let vec = match inner.model.embed(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[WS Ошибка] {}", e);
                        return;
                    }
                };
                let vec = core_lib::adapter::apply_if_loaded(&vec);
                let vec = core_lib::crypto::permute(&vec);
                if let Ok(results) = inner.index.search(&vec, 1) {
                    if let Some(r) = results.first() {
                        if r.distance < 0.05 {
                            return;
                        }
                    }
                }
                match inner.index.add(&text, &source, &vec) {
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
                    let inner = match s.inner.as_mut() {
                        Some(i) => i,
                        None => return,
                    };
                    let vec = match inner.model.embed(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("[Ошибка векторизации] {}", e);
                            return;
                        }
                    };
                    let vec = core_lib::adapter::apply_if_loaded(&vec);
                    let vec = core_lib::crypto::permute(&vec);
                    if let Ok(results) = inner.index.search(&vec, 1) {
                        if let Some(r) = results.first() {
                            if r.distance < 0.05 {
                                return;
                            }
                        }
                    }
                    let data_dir = inner.data_dir.clone();
                    match inner.index.add(&text, "clipboard", &vec) {
                        Ok(id) => {
                            println!("[Захват] id={} chars={}", id, text.chars().count());
                            let _ = inner.index.save(&data_dir);
                        }
                        Err(e) => eprintln!("[Ошибка индекса] {}", e),
                    }
                }) {
                    eprintln!("[Синапс] Ошибка: {:?}", e);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![unlock, search, feedback])
        .build(tauri::generate_context!())
        .expect("Tauri error")
        .run(move |app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(state) = app_handle.try_state::<Arc<Mutex<AppState>>>() {
                    let s = state.lock().unwrap();
                    if let Some(inner) = s.inner.as_ref() {
                        if let Err(e) = inner.index.save(&inner.data_dir) {
                            eprintln!("[Shutdown] Ошибка сохранения: {}", e);
                        } else {
                            println!("[Shutdown] Индекс сохранён.");
                        }
                    }
                }
            }
        });
}