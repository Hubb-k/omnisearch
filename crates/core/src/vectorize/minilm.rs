use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub struct MiniLM {
    session: ort::session::InMemorySession<'static>,
    tokenizer: Tokenizer,
}

impl MiniLM {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        ort::init().with_name("minilm").commit();

        let model_path =
            std::env::var("ORT_MODEL_PATH").unwrap_or_else(|_| "models/model.onnx".to_string());
        let tokenizer_path = std::env::var("ORT_TOKENIZER_PATH")
            .unwrap_or_else(|_| "models/tokenizer.json".to_string());

        let model_bytes = Box::leak(std::fs::read(&model_path)?.into_boxed_slice());
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_memory_directly(model_bytes)?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| format!("Tokenizer error: {}", e))?;

        Ok(Self { session, tokenizer })
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Encode error: {}", e))?;

        let len = encoding.get_ids().len();

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        // Сохраняем маску до передачи в тензор
        let mask_f32: Vec<f32> = mask.iter().map(|&x| x as f32).collect();

        let ids_tensor = Tensor::from_array(([1, len], ids))?;
        let mask_tensor = Tensor::from_array(([1, len], mask))?;
        let type_ids_tensor = Tensor::from_array(([1, len], type_ids))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids"      => ids_tensor,
            "attention_mask" => mask_tensor,
            "token_type_ids" => type_ids_tensor
        ])?;

        let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
        let seq_len = shape[1] as usize;
        let hidden = shape[2] as usize;

        let array = Array2::from_shape_vec((seq_len, hidden), data.to_vec())?;

        // Mean pooling с учётом attention mask
        let mask_sum: f32 = mask_f32.iter().sum::<f32>().max(1e-9);
        let mut pooled = vec![0.0f32; hidden];
        for token_idx in 0..seq_len {
            let token_mask = mask_f32[token_idx];
            for dim in 0..hidden {
                pooled[dim] += array[[token_idx, dim]] * token_mask;
            }
        }
        for dim in 0..hidden {
            pooled[dim] /= mask_sum;
        }

        // L2 нормализация
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = pooled.iter().map(|x| x / norm).collect();

        Ok(normalized)
    }
}
