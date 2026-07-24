#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::Path;

pub fn should_train(feedback_count: usize) -> bool {
    feedback_count > 0 && feedback_count % 50 == 0
}

pub fn spawn_finetune(data_dir: &str, model_dir: &str) {
    let data_dir = data_dir.to_string();
    let model_dir = model_dir.to_string();

    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(30));

        let script = Path::new(&model_dir).parent().unwrap()
            .join("scripts/finetune.py");

        if !script.exists() {
            eprintln!("[Train] finetune.py не найден");
            return;
        }

        eprintln!("[Train] Запуск fine-tune...");

        let mut cmd = std::process::Command::new("python");
        cmd.arg(&script)
            .arg("--data_dir").arg(&data_dir)
            .arg("--model_dir").arg(&model_dir)
            .env("OMP_NUM_THREADS", "1")
            .env("RAYON_NUM_THREADS", "1");

        #[cfg(windows)]
        cmd.creation_flags(0x00000040);

        match cmd.spawn() {
            Ok(mut child) => {
                let _ = child.wait();
                eprintln!("[Train] Fine-tune завершён");
            }
            Err(e) => eprintln!("[Train] Ошибка запуска: {}", e),
        }
    });
}