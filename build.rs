// build.rs — Compile CUDA kernels with nvcc when --features cuda is enabled
// Supports Windows native and WSL2 (auto-detects environment)

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let nvcc = which_nvcc();
    if let Some(nvcc) = nvcc {
        println!("cargo:warning=[m2m] Found nvcc: {}", nvcc);

        let cl_exe = find_msvc_cl();

        // Compile distance.cu to PTX
        let ptx_output = out_dir.join("distance.ptx");
        let mut cmd = Command::new(&nvcc);
        cmd.args([
            "--ptx",
            "--gpu-architecture=sm_86",
            "-O3",
            "--use_fast_math",
            "--extra-device-vectorization",
        ]);
        if let Some(ref cl) = cl_exe {
            cmd.args(["-ccbin", cl]);
        }
        cmd.args([
            &format!("--output-file={}", ptx_output.display()),
            "kernels/distance.cu",
        ]);

        let status = cmd.status().expect("Failed to run nvcc");
        if status.success() {
            println!("cargo:warning=[m2m] CUDA distance kernels compiled to PTX");
            println!("cargo:rerun-if-changed=kernels/distance.cu");
            println!("cargo:rustc-env=M2M_PTX_PATH={}", ptx_output.display());

            // Compile extended kernels
            let ext_ptx = out_dir.join("extended_kernels.ptx");
            let mut ext_cmd = Command::new(&nvcc);
            ext_cmd.args([
                "--ptx",
                "--gpu-architecture=sm_86",
                "-O3",
                "--use_fast_math",
                "--extra-device-vectorization",
            ]);
            if let Some(ref cl) = cl_exe {
                ext_cmd.args(["-ccbin", cl]);
            }
            ext_cmd.args([
                &format!("--output-file={}", ext_ptx.display()),
                "kernels/extended_kernels.cu",
            ]);
            if ext_cmd.status().map(|s| s.success()).unwrap_or(false) {
                println!("cargo:warning=[m2m] CUDA extended kernels compiled to PTX");
                println!("cargo:rerun-if-changed=kernels/extended_kernels.cu");
                println!("cargo:rustc-env=M2M_EXTENDED_PTX_PATH={}", ext_ptx.display());
            }
        } else {
            println!("cargo:warning=[m2m] nvcc compilation failed — GPU kernels unavailable");
        }
    } else {
        println!("cargo:warning=[m2m] nvcc not found — GPU kernels unavailable");
    }
}

fn which_nvcc() -> Option<String> {
    // 1. Check Linux PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout.lines().next().unwrap_or("").trim();
            if !line.is_empty() { return Some(line.to_string()); }
        }
    }
    // 2. Check CUDA_PATH env
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");
        if nvcc.exists() { return Some(nvcc.to_string_lossy().to_string()); }
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
        if nvcc.exists() { return Some(nvcc.to_string_lossy().to_string()); }
    }
    // 3. Check Windows CUDA toolkit (works from both WSL and Windows)
    for base in [
        PathBuf::from("/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        PathBuf::from("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"),
    ] {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let nvcc = entry.path().join("bin").join("nvcc.exe");
                if nvcc.exists() { return Some(nvcc.to_string_lossy().to_string()); }
            }
        }
    }
    None
}

fn find_msvc_cl() -> Option<String> {
    for base in [
        PathBuf::from("/mnt/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"),
        PathBuf::from("C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC"),
        PathBuf::from("/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"),
        PathBuf::from("C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC"),
    ] {
        if let Ok(entries) = std::fs::read_dir(&base) {
            for entry in entries.flatten() {
                let cl = entry.path().join("bin").join("Hostx64").join("x64").join("cl.exe");
                if cl.exists() { return Some(cl.to_string_lossy().to_string()); }
            }
        }
    }
    None
}
