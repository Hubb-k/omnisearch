use std::sync::OnceLock;

static PERMUTATION: OnceLock<[u16; 384]> = OnceLock::new();
static SEED_STORE: OnceLock<u64> = OnceLock::new();

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

fn hardware_seed() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    if let Ok(hostname) = std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")) {
        hostname.hash(&mut hasher);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .hash(&mut hasher);
    if let Ok(user) = std::env::var("USERNAME").or_else(|_| std::env::var("USER")) {
        user.hash(&mut hasher);
    }
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        home.hash(&mut hasher);
    }

    hasher.finish()
}

pub fn init() {
    let seed = hardware_seed();
    let _ = SEED_STORE.set(seed);
    let _ = PERMUTATION.set(build_permutation(seed));
    println!(
        "[Crypto] Карта перестановок инициализирована (seed={:#018x})",
        seed
    );
}

pub fn get_seed() -> u64 {
    *SEED_STORE.get().expect("Crypto not initialized")
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
