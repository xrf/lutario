fn main() {
    println!("cargo:rustc-link-lib=dylib=cblas");
    println!("cargo:rustc-link-lib=dylib=lapacke");
}
