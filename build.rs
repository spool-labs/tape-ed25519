//! Decide at build time whether the x86_64 vector backend is compiled in

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let is_x86_64 = matches!(
        std::env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("x86_64")
    );
    if is_x86_64 {
        println!("cargo:rustc-cfg=curve25519_backend=\"simd\"");
    }
}
