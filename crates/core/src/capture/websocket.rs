use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::collections::HashSet;
use tungstenite::accept;
use tungstenite::Message;
use serde::Deserialize;

#[derive(Deserialize)]
struct IncomingMessage {
    #[serde(rename = "type")]
    msg_type: String,
    text: String,
    source: String,
    title: Option<String>,
}

pub fn start_ws_server<F>(callback: F)
where
    F: Fn(String, String) + Send + Sync + 'static,
{
    let callback = Arc::new(callback);
    let seen_urls: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

    std::thread::spawn(move || {
        let listener = match TcpListener::bind("127.0.0.1:45678") {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[WebSocket] Не удалось запустить сервер: {}", e);
                return;
            }
        };

        println!("[WebSocket] Сервер запущен на ws://127.0.0.1:45678");

        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };

            let callback = callback.clone();
            let seen_urls = seen_urls.clone();

            std::thread::spawn(move || {
                let mut ws = match accept(stream) {
                    Ok(w) => w,
                    Err(e) => {
                        eprintln!("[WebSocket] Handshake error: {}", e);
                        return;
                    }
                };

                let mut current_url: Option<String> = None;

                loop {
                    match ws.read() {
                        Ok(Message::Text(text)) => {
                            if let Ok(msg) = serde_json::from_str::<IncomingMessage>(&text) {
                                if msg.msg_type == "index" && !msg.text.trim().is_empty() {
                                    // Первый чанк с новым URL — проверяем seen
                                    if current_url.as_deref() != Some(&msg.source) {
                                        let already_seen = {
                                            let mut seen = seen_urls.lock().unwrap();
                                            !seen.insert(msg.source.clone())
                                        };
                                        if already_seen {
                                            eprintln!("[WebSocket] Пропуск дубля: {}", msg.source);
                                            break;
                                        }
                                        current_url = Some(msg.source.clone());
                                    }

                                    let source = format!(
                                        "browser:{}",
                                        msg.title.unwrap_or_else(|| msg.source.clone())
                                    );
                                    callback(msg.text, source);
                                }
                            }
                        }
                        Ok(Message::Close(_)) | Err(_) => break,
                        _ => {}
                    }
                }
            });
        }
    });
}