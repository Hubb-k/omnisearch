use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

const DIM_IN: usize = 384;
const DIM_MID: usize = 128;
const ADAPTER_PATH: &str = "adapter.safetensors";
const BASE_THRESHOLD: usize = 7;

static ADAPTER: OnceLock<RwLock<Option<Adapter>>> = OnceLock::new();
static IS_TRAINING: AtomicBool = AtomicBool::new(false);
static LAST_TRAIN_SECS: AtomicU64 = AtomicU64::new(0);

struct TrainingGuard;

impl Drop for TrainingGuard {
    fn drop(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        LAST_TRAIN_SECS.store(now, Ordering::SeqCst);
        IS_TRAINING.store(false, Ordering::SeqCst);
    }
}

pub fn current_threshold() -> usize {
    let last = LAST_TRAIN_SECS.load(Ordering::SeqCst);
    if last == 0 {
        return BASE_THRESHOLD;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let elapsed_secs = now.saturating_sub(last);
    if elapsed_secs < 120 {
        BASE_THRESHOLD * 8
    } else if elapsed_secs < 300 {
        BASE_THRESHOLD * 4
    } else if elapsed_secs < 900 {
        BASE_THRESHOLD * 2
    } else {
        BASE_THRESHOLD
    }
}

pub struct Adapter {
    w1: Linear,
    w2: Linear,
    device: Device,
}

impl Adapter {
    fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        let w1 = linear(DIM_IN, DIM_MID, vb.pp("w1"))?;
        let w2 = linear(DIM_MID, DIM_IN, vb.pp("w2"))?;
        Ok(Self {
            w1,
            w2,
            device: Device::Cpu,
        })
    }

    pub fn apply(&self, vec: &[f32]) -> Vec<f32> {
        let t = match Tensor::from_slice(vec, (1, DIM_IN), &self.device) {
            Ok(t) => t,
            Err(_) => return vec.to_vec(),
        };
        let out = self
            .w1
            .forward(&t)
            .and_then(|x| x.relu())
            .and_then(|x| self.w2.forward(&x));
        match out {
            Ok(t) => t
                .flatten_all()
                .and_then(|t| t.to_vec1::<f32>())
                .unwrap_or_else(|_| vec.to_vec()),
            Err(_) => vec.to_vec(),
        }
    }
}

pub fn init(model_dir: &str) {
    let global = ADAPTER.get_or_init(|| RwLock::new(None));
    let path = Path::new(model_dir).join(ADAPTER_PATH);
    if path.exists() {
        match load(&path.to_string_lossy()) {
            Ok(adapter) => {
                if let Ok(mut w) = global.write() {
                    *w = Some(adapter);
                    eprintln!("[Adapter] Загружен из {}", path.display());
                }
            }
            Err(e) => eprintln!("[Adapter] Ошибка загрузки: {}", e),
        }
    } else {
        eprintln!("[Adapter] Файл не найден, работаем без адаптера.");
    }
}

pub fn apply_if_loaded(vec: &[f32]) -> Vec<f32> {
    let global = match ADAPTER.get() {
        Some(g) => g,
        None => return vec.to_vec(),
    };
    match global.read() {
        Ok(guard) => match guard.as_ref() {
            Some(adapter) => adapter.apply(vec),
            None => vec.to_vec(),
        },
        Err(_) => vec.to_vec(),
    }
}

fn load(path: &str) -> candle_core::Result<Adapter> {
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let adapter = Adapter::new(vb)?;
    varmap.load(path)?;
    Ok(adapter)
}

pub fn train(triplets: Vec<[Vec<f32>; 3]>, model_dir: String) {
    if triplets.is_empty() {
        return;
    }

    if IS_TRAINING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        eprintln!("[Adapter] Обучение уже запущено, пропуск.");
        return;
    }

    std::thread::spawn(move || {
        let _guard = TrainingGuard;

        eprintln!("[Adapter] Запуск обучения на {} триплетах...", triplets.len());

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let adapter = match Adapter::new(vb) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[Adapter] Ошибка инициализации: {}", e);
                return;
            }
        };

        let mut opt = match candle_nn::SGD::new(varmap.all_vars(), 0.01) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[Adapter] Ошибка оптимизатора: {}", e);
                return;
            }
        };

        for epoch in 0..10 {
            let mut total_loss = 0f32;

            for triplet in &triplets {
                let [anchor, positive, negative] = triplet;

                let a = match vec_to_tensor(anchor, &device) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let p = match vec_to_tensor(positive, &device) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let n = match vec_to_tensor(negative, &device) {
                    Ok(t) => t,
                    Err(_) => continue,
                };

                let loss = match triplet_loss(&adapter, &a, &p, &n, 0.2) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("[Adapter] Loss error: {}", e);
                        continue;
                    }
                };

                total_loss += loss.to_scalar::<f32>().unwrap_or(0.0);

                if let Err(e) = opt.backward_step(&loss) {
                    eprintln!("[Adapter] Backward error: {}", e);
                }
            }

            eprintln!(
                "[Adapter] Эпоха {}/10 loss={:.4}",
                epoch + 1,
                total_loss / triplets.len() as f32
            );
        }

        let out_path = Path::new(&model_dir).join(ADAPTER_PATH);
        if let Err(e) = varmap.save(&out_path) {
            eprintln!("[Adapter] Ошибка сохранения: {}", e);
            return;
        }
        eprintln!("[Adapter] Сохранён в {}", out_path.display());

        if let Some(global) = ADAPTER.get() {
            match load(&out_path.to_string_lossy()) {
                Ok(new_adapter) => {
                    if let Ok(mut w) = global.write() {
                        *w = Some(new_adapter);
                        eprintln!("[Adapter] Обновлён в runtime.");
                    }
                }
                Err(e) => eprintln!("[Adapter] Ошибка перезагрузки: {}", e),
            }
        }
    });
}

fn vec_to_tensor(v: &[f32], device: &Device) -> candle_core::Result<Tensor> {
    Tensor::from_slice(v, (1, DIM_IN), device)
}

fn triplet_loss(
    adapter: &Adapter,
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
) -> candle_core::Result<Tensor> {
    let device = &adapter.device;

    let a_out = adapter.w2.forward(&adapter.w1.forward(anchor)?.relu()?)?;
    let p_out = adapter.w2.forward(&adapter.w1.forward(positive)?.relu()?)?;
    let n_out = adapter.w2.forward(&adapter.w1.forward(negative)?.relu()?)?;

    let d_pos = cosine_distance(&a_out, &p_out)?;
    let d_neg = cosine_distance(&a_out, &n_out)?;

    let margin_t = Tensor::full(margin, d_pos.shape(), device)?;
    let zero = Tensor::zeros(d_pos.shape(), DType::F32, device)?;

    (d_pos - d_neg + margin_t)?.maximum(&zero)
}

fn cosine_distance(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let dot = (a * b)?.sum_all()?;
    let norm_a = a.sqr()?.sum_all()?.sqrt()?;
    let norm_b = b.sqr()?.sum_all()?.sqrt()?;
    let similarity = (dot / (norm_a * norm_b)?)?;
    let one = Tensor::ones(similarity.shape(), DType::F32, similarity.device())?;
    one - similarity
}