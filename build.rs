// build.rs — Compile CUDA kernels with nvcc when --features cuda is enabled
// Only runs when the `cuda` feature is active AND nvcc is found on PATH.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only compile CUDA kernels when the cuda feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Find nvcc
    let nvcc = which_nvcc();
    if let Some(nvcc) = nvcc {
        println!("cargo:warning=[m2m] Found nvcc: {}", nvcc);

        // Compile distance.cu to PTX
        let kernel_src = "kernels/distance.cu";
        let ptx_output = out_dir.join("distance.ptx");

        // Find MSVC cl.exe for nvcc host compiler
        let cl_exe = find_msvc_cl();
        let mut cmd = Command::new(&nvcc);
        cmd.args([
            "--ptx",
            "--gpu-architecture=sm_86",
            "--generate-code=arch=compute_86,code=[sm_86,compute_86]",
            "-O3",
            "--use_fast_math",
            "--extra-device-vectorization",
        ]);
        if let Some(ref cl) = cl_exe {
            cmd.args(["-ccbin", cl]);
        }
        cmd.args([
            &format!("--output-file={}", ptx_output.display()),
            kernel_src,
        ]);

        let status = cmd
            .status()
            .expect("Failed to run nvcc");

        if status.success() {
            println!("cargo:warning=[m2m] CUDA kernels compiled to PTX: {}", ptx_output.display());
            // Tell cargo to re-run if kernels change
            println!("cargo:rerun-if-changed=kernels/distance.cu");
            // Embed PTX path as env variable
            println!("cargo:rustc-env=M2M_PTX_PATH={}", ptx_output.display());
        } else {
            println!("cargo:warning=[m2m] nvcc compilation failed — GPU kernels unavailable");
        }
    } else {
        println!("cargo:warning=[m2m] nvcc not found — GPU kernels unavailable, using cudarc defaults");
    }
}

fn which_nvcc() -> Option<String> {
    // Check CUDA_PATH first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(cuda_path).join("bin").join("nvcc.exe");
        if nvcc.exists() {
            return Some(nvcc.to_string_lossy().to_string());
        }
    }
    // Check PATH
    if let Ok(output) = Command::new("where").arg("nvcc.exe").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout);
            let line = path.lines().next().unwrap_or("").trim();
            if !line.is_empty() {
                return Some(line.to_string());
            }
        }
    }
    None
}

fn find_msvc_cl() -> Option<String> {
    // Search BuildTools for cl.exe
    let base = PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC");
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            let cl = entry.path().join("bin").join("Hostx64").join("x64").join("cl.exe");
            if cl.exists() {
                return Some(cl.to_string_lossy().to_string());
            }
        }
    }
    // Also check Community edition
    let base2 = PathBuf::from(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC");
    if let Ok(entries) = std::fs::read_dir(&base2) {
        for entry in entries.flatten() {
            let cl = entry.path().join("bin").join("Hostx64").join("x64").join("cl.exe");
            if cl.exists() {
                return Some(cl.to_string_lossy().to_string());
            }
        }
    }
    None
}
