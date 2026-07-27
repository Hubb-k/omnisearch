use argon2::{password_hash::SaltString, Argon2, PasswordHasher};
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use rand::{rngs::OsRng, RngCore};
use std::path::Path;
use std::sync::OnceLock;

static PERMUTATION: OnceLock<[u16; 384]> = OnceLock::new();
static SEED_STORE: OnceLock<u64> = OnceLock::new();
static MASTER_KEY: OnceLock<[u8; 32]> = OnceLock::new();

const KEY_FILE: &str = "key.bin";
const SALT_FILE: &str = "key.salt";
const NONCE_LEN: usize = 12;

fn build_permutation(seed: u64) -> [u16; 384] {
    let mut state = seed;
    let mut indices: [u16; 384] = std::array::from_fn(|i| i as u16);
    for i in (1..384).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn seed_from_master_key(key: &[u8; 32]) -> u64 {
    let mut seed = 0u64;
    for chunk in key.chunks(8) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        seed ^= u64::from_le_bytes(bytes);
    }
    seed
}

fn derive_key(password: &str, salt_bytes: &[u8; 32]) -> [u8; 32] {
    let salt_b64 = STANDARD_NO_PAD.encode(salt_bytes);
    let salt = SaltString::from_b64(&salt_b64).expect("Invalid salt");
    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .expect("Argon2 failed");
    let hash_bytes = hash.hash.expect("No hash output");
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash_bytes.as_bytes()[..32]);
    out
}

fn encrypt_master_key(master_key: &[u8; 32], derived_key: &[u8; 32]) -> Vec<u8> {
    let cipher = ChaCha20Poly1305::new(derived_key.into());
    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, master_key.as_ref())
        .expect("Encrypt failed");
    let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

fn decrypt_master_key(blob: &[u8], derived_key: &[u8; 32]) -> Option<[u8; 32]> {
    if blob.len() < NONCE_LEN {
        return None;
    }
    let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
    let cipher = ChaCha20Poly1305::new(derived_key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher.decrypt(nonce, ciphertext).ok()?;
    if plaintext.len() != 32 {
        return None;
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&plaintext);
    Some(key)
}

pub fn init_with_password(data_dir: &str, password: &str) -> Result<(), String> {
    let key_path = Path::new(data_dir).join(KEY_FILE);
    let salt_path = Path::new(data_dir).join(SALT_FILE);

    let master_key: [u8; 32] = if key_path.exists() && salt_path.exists() {
        let salt_bytes: [u8; 32] = std::fs::read(&salt_path)
            .map_err(|e| format!("Не удалось прочитать соль: {}", e))?
            .try_into()
            .map_err(|_| "Неверный размер соли".to_string())?;
        let derived = derive_key(password, &salt_bytes);
        let blob =
            std::fs::read(&key_path).map_err(|e| format!("Не удалось прочитать key.bin: {}", e))?;
        decrypt_master_key(&blob, &derived).ok_or_else(|| "Неверный пароль".to_string())?
    } else {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| format!("Не удалось создать data_dir: {}", e))?;
        let mut salt_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut salt_bytes);
        let mut new_master_key = [0u8; 32];
        OsRng.fill_bytes(&mut new_master_key);
        let derived = derive_key(password, &salt_bytes);
        let blob = encrypt_master_key(&new_master_key, &derived);
        std::fs::write(&salt_path, salt_bytes)
            .map_err(|e| format!("Не удалось записать соль: {}", e))?;
        std::fs::write(&key_path, &blob)
            .map_err(|e| format!("Не удалось записать key.bin: {}", e))?;
        eprintln!("[Crypto] Новый мастер-ключ создан и сохранён.");
        new_master_key
    };

    let seed = seed_from_master_key(&master_key);
    let _ = MASTER_KEY.set(master_key);
    let _ = SEED_STORE.set(seed);
    let _ = PERMUTATION.set(build_permutation(seed));
    eprintln!("[Crypto] Инициализирован (seed={:#018x})", seed);
    Ok(())
}

pub fn get_seed() -> u64 {
    *SEED_STORE.get().expect("Crypto not initialized")
}

pub fn get_db_key() -> String {
    let key = MASTER_KEY.get().expect("Crypto not initialized");
    key.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn encrypt_blob(data: &[u8]) -> Vec<u8> {
    let key = MASTER_KEY.get().expect("Crypto not initialized");
    let cipher = ChaCha20Poly1305::new(key.into());
    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher.encrypt(nonce, data).expect("Encrypt failed");
    let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    out
}

pub fn decrypt_blob(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < NONCE_LEN {
        return Err("Блоб слишком короткий".to_string());
    }
    let key = MASTER_KEY.get().expect("Crypto not initialized");
    let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
    let cipher = ChaCha20Poly1305::new(key.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| "Неверный ключ или повреждён файл".to_string())
}

pub fn permute(v_std: &[f32]) -> Vec<f32> {
    let map = PERMUTATION.get().expect("Crypto not initialized");
    let mut v_local = vec![0.0f32; 384];
    for i in 0..384 {
        v_local[map[i] as usize] = v_std[i];
    }
    v_local
}

pub fn unpermute(v_local: &[f32]) -> Vec<f32> {
    let map = PERMUTATION.get().expect("Crypto not initialized");
    let mut v_std = vec![0.0f32; 384];
    for i in 0..384 {
        v_std[i] = v_local[map[i] as usize];
    }
    v_std
}
