# OmniSearch — Net Intelligence

A local semantic search engine that indexes everything you read, copy, and browse. No data leaves your device.

## What it does

OmniSearch is a native desktop application built with Rust and Tauri. It captures your digital context in real time and makes it searchable by meaning:

- **Clipboard** — everything you copy
- **Browser** — pages you read (Chrome extension via WebSocket)
- **Files** — bulk import of text documents

Search by meaning with `Alt+Space`. Everything runs locally, everything is encrypted.

## Tech stack

| Component | Technology |
|---|---|
| Language | Rust |
| UI | Tauri 2 |
| Vectorization | MiniLM-L12 multilingual (ONNX Runtime) |
| Vector index | usearch HNSW |
| Full-text search | SQLite FTS5 |
| Database encryption | SQLCipher AES-256 |
| Vector obfuscation | DAP (Dynamic Axis Permutation) |

## Security

Two layers of on-disk data protection:

- `vectors.usearch` — vector axes shuffled by hardware fingerprint (DAP)
- `meta.sqlite` — transparent AES-256 page encryption via SQLCipher

The encryption key is derived from hardware descriptors of your machine at runtime. Moving the files to another device renders them unreadable noise.

## Requirements

- Rust stable
- OpenSSL 4.x Win64 (installed to `C:/Program Files/OpenSSL-Win64`)
- ONNX Runtime native library (place in `libs/onnxruntime/`)
- MiniLM multilingual model (place in `models/`)

## Setup

### 1. Download the model

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    allow_patterns=['onnx/model.onnx', 'tokenizer.json'],
    local_dir='./models_tmp'
)
"
cp models_tmp/onnx/model.onnx models/model.onnx
cp models_tmp/tokenizer.json models/tokenizer.json
```

### 2. Download ONNX Runtime

Download `onnxruntime-win-x64-*.zip` from [GitHub Releases](https://github.com/microsoft/onnxruntime/releases), extract and place:

```
libs/
└── onnxruntime/
    └── lib/
        ├── onnxruntime.lib
        └── onnxruntime.dll
```

### 3. Configure OpenSSL paths

Add to `.cargo/config.toml`:

```toml
[env]
OPENSSL_LIB_DIR = "C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD"
OPENSSL_INCLUDE_DIR = "C:/Program Files/OpenSSL-Win64/include"

[target.x86_64-pc-windows-msvc]
rustflags = [
    "-L", "libs/onnxruntime/lib",
    "-L", "C:/Program Files/OpenSSL-Win64/lib/VC/x64/MD",
]
```

### 4. Copy DLLs

After a clean build, copy the required DLLs to `target/debug/`:

```bash
cp libs/onnxruntime/lib/onnxruntime.dll target/debug/
cp "C:/Program Files/OpenSSL-Win64/bin/libcrypto-4-x64.dll" target/debug/
cp "C:/Program Files/OpenSSL-Win64/bin/libssl-4-x64.dll" target/debug/
```

### 5. Run

```bash
cargo tauri dev
```

## Browser Extension

Load `apps/browser-ext/` as an unpacked extension in Chrome:

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `apps/browser-ext/` folder

The extension captures page text and sends it to the local WebSocket server on port `45678`.

## Bulk Import

Index a folder of `.txt` files:

```bash
cargo run -p importer -- "/path/to/folder"
```

## Architecture

```
Clipboard listener     → embed → permute → HNSW + SQLite
Browser (WS :45678)    → embed → permute → HNSW + SQLite
Bulk importer          → embed → permute → HNSW + SQLite

Search (Alt+Space):
  Query → embed → permute → HNSW + FTS5 → ranked results
  Auto-feedback: similarity >= 0.78 → logged as training signal
  Manual feedback: click on result → logged
  Fine-tune: every 50 feedback pairs → background model update
```

## Roadmap

- **Phase 1 (current):** Local semantic search, clipboard + browser capture, encrypted storage, feedback loop
- **Phase 2:** P2P node network, gradient sharing, `$RAW` contribution token

## License

MIT
