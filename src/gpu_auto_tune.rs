//! GPU Auto-Tuning and optimization for SplatsDB.
//! Detects hardware, benchmarks performance, configures optimal parameters.
//! Ported from splatsdb Python.

use std::collections::HashMap;

/// Detected GPU profile.
#[derive(Debug, Clone)]
pub struct GpuProfile {
    pub vendor: String,
    pub device_name: String,
    pub vram_mb: usize,
    pub compute_units: usize,
    pub max_workgroup_size: usize,
    pub optimal_batch_size: usize,
    pub memory_bandwidth_gbps: f64,
    pub supports_fp16: bool,
    pub supports_subgroups: bool,
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self {
            vendor: "Unknown".into(),
            device_name: "CPU Fallback".into(),
            vram_mb: 0,
            compute_units: 1,
            max_workgroup_size: 256,
            optimal_batch_size: 32,
            memory_bandwidth_gbps: 0.0,
            supports_fp16: false,
            supports_subgroups: false,
        }
    }
}

/// Benchmark result for a specific configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub ops_per_sec: f64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub memory_used_mb: f64,
}

/// Tuning parameters.
#[derive(Debug, Clone)]
pub struct TuningParams {
    pub batch_size: usize,
    pub workgroup_size: usize,
    pub chunk_size: usize,
    pub use_fp16: bool,
    pub use_subgroups: bool,
}

impl Default for TuningParams {
    fn default() -> Self {
        Self {
            batch_size: 32,
            workgroup_size: 256,
            chunk_size: 8192,
            use_fp16: false,
            use_subgroups: false,
        }
    }
}

/// GPU auto-tuner: detects hardware, benchmarks, and configures optimal parameters.
pub struct GpuAutoTuner {
    profile: Option<GpuProfile>,
    benchmarks: HashMap<String, BenchmarkResult>,
    tuning_params: TuningParams,
}

impl GpuAutoTuner {
    /// New.
    pub fn new() -> Self {
        Self {
            profile: None,
            benchmarks: HashMap::new(),
            tuning_params: TuningParams::default(),
        }
    }

    /// Detect GPU and create profile. Returns CPU profile if no GPU found.
    ///
    /// Wired to `crate::gpu::gpu_info()` — previously this always returned the
    /// hardcoded "CPU Fallback" profile even on CUDA machines.
    pub fn detect_gpu(&mut self) -> &GpuProfile {
        let profile = match crate::gpu::gpu_info() {
            Some(info) => GpuProfile {
                vendor: "NVIDIA".into(),
                device_name: info.clone(),
                vram_mb: parse_total_vram_mb(&info),
                // fp16 needs compute capability >= 6.0 (sm_60)
                supports_fp16: parse_sm_arch(&info) >= 60,
                // compute_units / bandwidth are not exposed by gpu_info();
                // keep the safe defaults rather than inventing numbers.
                ..GpuProfile::default()
            },
            None => GpuProfile::default(),
        };
        self.profile = Some(profile);
        self.profile.as_ref().expect("profile was just set")
    }

    /// Set profile manually (e.g., from config).
    pub fn set_profile(&mut self, profile: GpuProfile) {
        self.profile = Some(profile);
    }

    /// Run a benchmark with given parameters, return ops/sec and latency.
    pub fn benchmark(
        &mut self,
        name: &str,
        n_ops: usize,
        f: impl Fn() -> std::time::Duration,
    ) -> &BenchmarkResult {
        let mut latencies = Vec::with_capacity(n_ops);
        let mut total_time = std::time::Duration::ZERO;

        // Single clock: the duration reported by `f` feeds BOTH the per-op
        // latencies and total_time (previously a second Instant::elapsed()
        // re-measured the same op and could disagree with `f`).
        for _ in 0..n_ops {
            let elapsed = f();
            latencies.push(elapsed.as_secs_f64() * 1000.0);
            total_time += elapsed;
        }

        // n_ops == 0: latencies is empty and indexing panics — return zeros.
        if latencies.is_empty() {
            let result = BenchmarkResult {
                name: name.to_string(),
                ops_per_sec: 0.0,
                avg_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                memory_used_mb: 0.0,
            };
            self.benchmarks.insert(name.to_string(), result);
            return self
                .benchmarks
                .get(name)
                .expect("benchmark was just inserted");
        }

        latencies.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("f64 latencies should be comparable")
        });

        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        // Clamp the p99 index to the last element: floor(len * 0.99) can hit
        // `len` at rounding boundaries and panic on out-of-bounds access.
        let p99_idx = ((latencies.len() as f64 * 0.99) as usize).min(latencies.len() - 1);
        let p99_latency = latencies[p99_idx];
        let total_secs = total_time.as_secs_f64();
        let ops_per_sec = if total_secs > 0.0 {
            n_ops as f64 / total_secs
        } else {
            0.0
        };

        let result = BenchmarkResult {
            name: name.to_string(),
            ops_per_sec,
            avg_latency_ms: avg_latency,
            p99_latency_ms: p99_latency,
            memory_used_mb: 0.0,
        };

        self.benchmarks.insert(name.to_string(), result);
        self.benchmarks
            .get(name)
            .expect("benchmark was just inserted")
    }

    /// Auto-tune batch size by trying different values.
    pub fn auto_tune_batch_size(
        &mut self,
        sizes: &[usize],
        f: impl Fn(usize) -> std::time::Duration,
    ) -> usize {
        let mut best_size = sizes[0];
        let mut best_ops = f64::MIN;

        for &size in sizes {
            let duration = f(size);
            let ops = size as f64 / duration.as_secs_f64();
            if ops > best_ops {
                best_ops = ops;
                best_size = size;
            }
        }

        self.tuning_params.batch_size = best_size;
        best_size
    }

    /// Auto-tune workgroup size.
    pub fn auto_tune_workgroup_size(
        &mut self,
        sizes: &[usize],
        f: impl Fn(usize) -> std::time::Duration,
    ) -> usize {
        let mut best_size = sizes[0];
        let mut best_ops = f64::MIN;

        for &size in sizes {
            let duration = f(size);
            let ops = 1.0 / duration.as_secs_f64();
            if ops > best_ops {
                best_ops = ops;
                best_size = size;
            }
        }

        self.tuning_params.workgroup_size = best_size;
        best_size
    }

    /// Get current tuning parameters.
    pub fn tuning_params(&self) -> &TuningParams {
        &self.tuning_params
    }

    /// Get a benchmark result by name.
    pub fn get_benchmark(&self, name: &str) -> Option<&BenchmarkResult> {
        self.benchmarks.get(name)
    }

    /// Get GPU profile.
    pub fn profile(&self) -> Option<&GpuProfile> {
        self.profile.as_ref()
    }

    /// Calculate optimal chunk size based on available memory.
    pub fn optimal_chunk_size(&self, dim: usize, available_mb: f64) -> usize {
        // chunk_size * dim * 4 bytes < available_memory
        let max_bytes = (available_mb * 1024.0 * 1024.0) as usize;
        let max_per_chunk = max_bytes / (dim * 4).max(1);
        // Round down to power of 2
        let mut chunk = max_per_chunk;
        chunk = chunk.next_power_of_two() / 2;
        chunk.clamp(256, 8192)
    }
}

impl Default for GpuAutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse total VRAM (MB) out of a `gpu_info()` string, which is formatted as
/// "<name> (sm_{}{}), {:.1} GB total, {:.1} GB free".
fn parse_total_vram_mb(info: &str) -> usize {
    info.split(", ")
        .find(|part| part.ends_with("GB total"))
        .and_then(|part| part.trim_end_matches("GB total").trim().parse::<f64>().ok())
        .map(|gb| (gb * 1024.0) as usize)
        .unwrap_or(0)
}

/// Parse the SM architecture number (e.g. 86 for "sm_86") out of a
/// `gpu_info()` string. 0 when absent/unparseable.
fn parse_sm_arch(info: &str) -> u32 {
    info.split_once("(sm_")
        .and_then(|(_, rest)| {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse().ok()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gpu() {
        let mut tuner = GpuAutoTuner::new();
        let profile = tuner.detect_gpu();
        // GPU-agnostic: with a CUDA device detect_gpu must report it; without
        // one it must return the CPU fallback profile (no GPU needed to run).
        match crate::gpu::gpu_info() {
            Some(info) => {
                assert_eq!(profile.vendor, "NVIDIA");
                assert!(profile.vram_mb > 0, "vram must parse from: {info}");
                assert_eq!(profile.device_name, info);
            }
            None => {
                assert_eq!(profile.vendor, "Unknown");
                assert_eq!(profile.device_name, "CPU Fallback");
            }
        }
    }

    #[test]
    fn test_parse_gpu_info_fields() {
        let info = "NVIDIA GeForce RTX 3090 (sm_86), 24.0 GB total, 23.3 GB free";
        assert_eq!(parse_total_vram_mb(info), 24576);
        assert_eq!(parse_sm_arch(info), 86);
        assert_eq!(parse_sm_arch("CPU Fallback"), 0);
        assert_eq!(parse_total_vram_mb("no vram info here"), 0);
    }

    #[test]
    fn test_benchmark() {
        let mut tuner = GpuAutoTuner::new();
        let result = tuner.benchmark("test", 100, || {
            std::thread::sleep(std::time::Duration::from_micros(10));
            std::time::Duration::from_micros(10)
        });
        assert!(result.ops_per_sec > 0.0);
        assert!(result.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_auto_tune_batch() {
        let mut tuner = GpuAutoTuner::new();
        let best = tuner.auto_tune_batch_size(&[1, 10, 100], |size| {
            // Simulate: larger batches are faster per-item
            std::time::Duration::from_micros((1000 / size) as u64)
        });
        assert_eq!(best, 100);
    }

    #[test]
    fn test_optimal_chunk_size() {
        let tuner = GpuAutoTuner::new();
        let chunk = tuner.optimal_chunk_size(640, 100.0); // 100MB available
        assert!(chunk >= 256);
        assert!(chunk <= 8192);
    }
}
