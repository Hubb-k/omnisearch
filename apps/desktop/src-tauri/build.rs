fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let lib_path = format!("{}/../../../libs/onnxruntime/lib", root);
    
    println!("cargo:rustc-link-search=native={}", lib_path);
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-env=ORT_LIB_PATH={}", lib_path);
    println!("cargo:rerun-if-changed=build.rs");
    
    tauri_build::build()
}