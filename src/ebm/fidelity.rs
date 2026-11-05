//! Epps–Pulley distributional-fidelity diagnostics for splat models.
//!
//! Measures how far an embedding batch is from an isotropic Gaussian (the
//! LeJEPA/SIGReg target, arXiv:2511.08544) and whether a splat mixture
//! preserves the distributional statistics that the test sees.
//!
//! Methodology ported from the validated cross-experiment of 2026-11-05
//! (20 Newsgroups × all-MiniLM-L6-v2; receipt JSONs committed alongside the
//! docs, see docs/EP_FIDELITY.md):
//!
//! * H1 confirmed: distance-to-isotropy correlates positively with one-class
//!   AUROC (Spearman ρ=0.5023, p=0.024); pushing embeddings toward N(0,I)
//!   degrades detection to chance (0.956 → 0.500 at λ=1). Isotropy is NOT a
//!   universal optimum — it is probe-risk-specific.
//! * H2 refuted: vMF mixture samples score ~60× closer to the Gaussian than
//!   the real data (median gap ratio 0.016): the splat model compresses
//!   enough to rank (AUROC 0.949–0.991) but not to reproduce the
//!   distribution. This module quantifies that gap so a future refit
//!   (learned per-cluster κ/α instead of counts/const) can be measured.
//!
//! **A5 constraint (docs/A5_FINDINGS.md): this is a diagnostic metric only.
//! Never feed it (or any density/energy signal) into the retrieval ranking
//! score — energy-as-score degrades recall@10 from 0.90 to ~0.00.**
//!
//! Statistic (SIGReg-style sketch of the Epps–Pulley test):
//! * assume rows of the sample are L2-normalized (sphere S^{d-1});
//! * draw `n_directions` uniform random unit directions;
//! * project: `u_i = sqrt(d) * (a^T z_i)` — standard deviation 1 under the
//!   isotropic null;
//! * per direction, compare the empirical characteristic function against
//!   the N(0,1) one on a quadrature grid with Gaussian window:
//!   `EP = Σ_t w(t) |φ̂(t) − e^{−t²/2}|² Δt`, `w(t) = e^{−t²/2}`;
//! * report the mean over directions (no factor N: comparable across
//!   batch sizes).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Beta, Normal, Uniform};

/// Configuration of the sketched Epps–Pulley statistic.
#[derive(Debug, Clone)]
pub struct EpConfig {
    /// Random unit directions to sketch with (LeJEPA trains with 256; 64 is
    /// enough for a stable diagnostic — validated in the 2026-11-05 run).
    pub n_directions: usize,
    /// Lower end of the quadrature grid for t.
    pub t_min: f64,
    /// Upper end of the quadrature grid for t.
    pub t_max: f64,
    /// Number of quadrature points on `[t_min, t_max]`.
    pub grid_points: usize,
    /// Master seed — deterministic by contract.
    pub seed: u64,
}

impl Default for EpConfig {
    fn default() -> Self {
        EpConfig {
            n_directions: 64,
            t_min: -5.0,
            t_max: 5.0,
            grid_points: 65,
            seed: 42,
        }
    }
}

/// Result of a fidelity comparison between real data and splat samples.
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityReport {
    /// EP distance-to-isotropy of the real batch.
    pub ep_real: f64,
    /// EP distance-to-isotropy of samples drawn from the splat mixture.
    pub ep_synth: f64,
    /// `ep_synth / ep_real` — 1.0 means the mixture reproduces the
    /// statistics the test sees; << 1 means the model is smoother than the
    /// data (measured 0.016 median in the 2026-11-05 experiment).
    pub gap_ratio: f64,
}

fn quadrature_grid(cfg: &EpConfig) -> (Vec<f64>, f64) {
    let n = cfg.grid_points.max(2);
    let dt = (cfg.t_max - cfg.t_min) / (n - 1) as f64;
    let t: Vec<f64> = (0..n).map(|i| cfg.t_min + i as f64 * dt).collect();
    (t, dt)
}

/// Uniform random unit directions, shape `(n_directions, dim)`.
///
/// Deterministic for a given `(dim, n_directions, seed)`.
pub fn random_directions(dim: usize, n_directions: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).expect("fixed sigma");
    let mut dirs = Array2::<f32>::zeros((n_directions, dim));
    for mut row in dirs.rows_mut() {
        loop {
            for v in row.iter_mut() {
                *v = rng.sample(normal);
            }
            let norm = row.dot(&row).sqrt();
            if norm > 1e-6 {
                row.mapv_inplace(|v| v / norm);
                break;
            }
        }
    }
    dirs
}

/// Sketched Epps–Pulley distance-to-isotropy of an L2-normalized batch.
///
/// `samples` has shape `(n, dim)` with unit-norm rows. Returns the mean
/// per-direction statistic. Cost: `O(n * dim * n_directions)`.
pub fn ep_statistic(samples: ArrayView2<f32>, cfg: &EpConfig) -> f64 {
    let n = samples.nrows();
    let dim = samples.ncols();
    assert!(n >= 2, "need at least 2 samples");
    let dirs = random_directions(dim, cfg.n_directions, cfg.seed);

    // Projections: (n, m) = samples (n, d) × dirs^T (d, m), standardized.
    let proj = samples.dot(&dirs.t()) * (dim as f32).sqrt();

    let (t, dt) = quadrature_grid(cfg);
    let mut total = 0.0_f64;
    for col in proj.columns() {
        let mut stat = 0.0_f64;
        for &tv in t.iter() {
            // Empirical characteristic function at t (mean of e^{i t u}).
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for &u in col.iter() {
                let tu = tv * u as f64;
                re += tu.cos();
                im += tu.sin();
            }
            re /= n as f64;
            im /= n as f64;
            // Target: CF of N(0,1) is e^{-t²/2} (purely real).
            let target = (-0.5 * tv * tv).exp();
            let diff_re = re - target;
            let w = (-0.5 * tv * tv).exp();
            stat += w * (diff_re * diff_re + im * im);
        }
        total += stat * dt;
    }
    total / cfg.n_directions as f64
}

/// Theoretical mean resultant length `A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ)`
/// via the stable continued fraction
/// `A_ν = 1 / (2ν/κ + A_{ν+1})`, seeded with the large-ν asymptotic
/// `A ≈ κ/(2ν)`. (Direct `scipy`-style Bessel ratios underflow at high d.)
pub fn vmf_mean_resultant(dim: usize, kappa: f64) -> f64 {
    let nu = dim as f64 / 2.0;
    let start = nu + 4096.0;
    let mut a = kappa / (2.0 * start);
    let mut m = start - 1.0;
    while m >= nu - 0.5 {
        a = 1.0 / (2.0 * m / kappa + a);
        m -= 1.0;
    }
    a
}

/// Draw one sample from vMF(mean `mu`, concentration `kappa`) on the unit
/// sphere (Wood 1994 accept-reject, as popularized by spherecluster):
/// accept when `κ·w + (d−1)·ln(1−x0·w) − c ≥ ln u`, tangent direction from
/// an N(0,I) vector projected onto the orthogonal complement of `mu`.
pub fn sample_vmf<R: Rng + ?Sized>(rng: &mut R, mu: ArrayView1<f32>, kappa: f64) -> Array1<f32> {
    let d = mu.len();
    assert!(d >= 2, "vMF needs dim >= 2");
    assert!(kappa > 0.0, "kappa must be positive");
    let df = (d - 1) as f64;
    let b = (-2.0 * kappa + (4.0 * kappa * kappa + df * df).sqrt()) / df;
    let x0 = (1.0 - b) / (1.0 + b);
    let c = kappa * x0 + df * (1.0 - x0 * x0).ln();

    let e1: Array1<f32> = {
        let norm = mu.dot(&mu).sqrt().max(1e-12);
        mu.mapv(|v| v / norm)
    };
    let beta = Beta::new(df / 2.0, df / 2.0).expect("valid beta params");
    let log_u_dist = Uniform::new(f64::EPSILON, 1.0);

    for _ in 0..10_000 {
        let z: f64 = rng.sample(beta);
        let w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z);
        let accept = kappa * w + df * (1.0 - x0 * w).ln() - c;
        if accept >= rng.sample(log_u_dist).ln() {
            // Tangent vector at e1, normalized; sample = w·e1 + √(1−w²)·t.
            let normal = Normal::new(0.0_f32, 1.0).expect("fixed sigma");
            let mut t: Array1<f32> = (0..d).map(|_| rng.sample(normal)).collect();
            let dot = t.dot(&e1);
            for (tv, &ev) in t.iter_mut().zip(e1.iter()) {
                *tv -= dot * ev;
            }
            let tn = t.dot(&t).sqrt().max(1e-12);
            let s = ((1.0 - w * w).max(0.0).sqrt()) as f32;
            let mut x = Array1::<f32>::zeros(d);
            for i in 0..d {
                x[i] = w as f32 * e1[i] + s * t[i] / tn as f32;
            }
            let xn = x.dot(&x).sqrt().max(1e-12);
            x.mapv_inplace(|v| v / xn);
            return x;
        }
    }
    panic!("vMF sampler failed to accept in 10k iterations (d={d}, kappa={kappa})");
}

fn sample_mixture(
    rng: &mut ChaCha8Rng,
    mu: ArrayView2<f32>,
    kappa: &[f64],
    n_samples: usize,
) -> Array2<f32> {
    assert_eq!(mu.nrows(), kappa.len(), "one kappa per splat");
    let total: f64 = kappa.iter().sum();
    assert!(total > 0.0, "kappas must sum > 0");
    // Component prior = κ normalized, as in the 2026-11-05 experiment.
    let cum: Vec<f64> = {
        let mut acc = 0.0;
        kappa
            .iter()
            .map(|&k| {
                acc += k / total;
                acc
            })
            .collect()
    };
    let unif = Uniform::new(0.0_f64, 1.0);
    let dim = mu.ncols();
    let mut out = Array2::<f32>::zeros((n_samples, dim));
    for mut row in out.rows_mut() {
        let pick: f64 = rng.sample(unif);
        let comp = cum
            .iter()
            .position(|&c| c >= pick)
            .unwrap_or(kappa.len() - 1);
        let s = sample_vmf(rng, mu.row(comp), kappa[comp]);
        row.assign(&s);
    }
    out
}

/// Compare the real batch against samples of its splat mixture.
///
/// * `real` — `(n, dim)` L2-normalized embeddings (the data the splats were
///   fitted on, e.g. k-means centroids with κ = cluster sizes);
/// * `mu` — `(k, dim)` splat centers, unit norm;
/// * `kappa` — per-splat concentrations (cluster counts in the standard fit).
///
/// Deterministic for a given `cfg.seed`. Diagnostic only — see the A5 note
/// in the module docs.
pub fn fidelity_gap_ratio(
    real: ArrayView2<f32>,
    mu: ArrayView2<f32>,
    kappa: &[f64],
    cfg: &EpConfig,
) -> FidelityReport {
    let ep_real = ep_statistic(real, cfg);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed.wrapping_add(1));
    let synth = sample_mixture(&mut rng, mu, kappa, real.nrows());
    let ep_synth = ep_statistic(synth.view(), cfg);
    let ratio = ep_synth / ep_real.max(1e-12);
    FidelityReport {
        ep_real,
        ep_synth,
        gap_ratio: ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn normalized_batch(rows: &[(usize, f64)]) -> Array2<f32> {
        // rows: (dim index to bump, bump size) — deterministic clustered data
        let mut a = Array2::<f32>::zeros((rows.len(), 8));
        for (i, &(bump_idx, bump)) in rows.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(1_000 + i as u64);
            let normal = Normal::new(0.0_f32, 0.05).unwrap();
            for v in a.row_mut(i).iter_mut() {
                *v = rng.sample(normal);
            }
            a[[i, bump_idx]] += bump as f32;
            let norm = a.row(i).dot(&a.row(i)).sqrt();
            a.row_mut(i).mapv_inplace(|v| v / norm);
        }
        a
    }

    #[test]
    fn ep_discriminates_gaussian_from_skewed() {
        // Receipt (Python, 2026-11-05): ratio 1365x. Rust assert: > 10x.
        let cfg = EpConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let gauss: Vec<f32> = (0..2000).map(|_| rng.sample(normal)).collect();
        let unif = Uniform::new(0.2_f64, 1.0);
        let skew: Vec<f32> = (0..2000).map(|_| (rng.sample(unif).ln()) as f32).collect();
        let g = Array2::from_shape_vec((2000, 1), gauss).unwrap();
        let s = Array2::from_shape_vec((2000, 1), skew).unwrap();
        let ep_g = ep_statistic(g.view(), &cfg);
        let ep_s = ep_statistic(s.view(), &cfg);
        assert!(ep_g < 0.01, "gaussian EP should be tiny, got {ep_g}");
        assert!(ep_s > 0.1, "skewed EP should be large, got {ep_s}");
        assert!(ep_s / ep_g > 10.0, "ratio {}", ep_s / ep_g);
    }

    #[test]
    fn ep_is_deterministic() {
        let cfg = EpConfig::default();
        let batch = normalized_batch(&[(0, 1.0), (4, 1.2), (2, 0.8)]);
        let a = ep_statistic(batch.view(), &cfg);
        let b = ep_statistic(batch.view(), &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn vmf_mean_resultant_matches_exact_3d() {
        // A_3(10) = coth(10) − 1/10 = 0.9 exactly.
        let a = vmf_mean_resultant(3, 10.0);
        assert!((a - 0.9).abs() < 1e-6, "got {a}");
        // Receipts (Python continued fraction): d=64, κ=5 → 0.0777;
        // d=384, κ=20 → 0.0519.
        assert!((vmf_mean_resultant(64, 5.0) - 0.0777).abs() < 1e-3);
        assert!((vmf_mean_resultant(384, 20.0) - 0.0519).abs() < 1e-3);
    }

    #[test]
    fn vmf_sampler_reproduces_resultant() {
        // Receipt (Python, n=500): |R_emp − R_theo| < 0.03. n=300, tol 0.05.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let dim = 384;
        let kappa = 20.0;
        let mut mu = Array1::<f32>::zeros(dim);
        mu[0] = 1.0;
        let n = 300;
        let mut mean = Array1::<f64>::zeros(dim);
        for _ in 0..n {
            let x = sample_vmf(&mut rng, mu.view(), kappa);
            for i in 0..dim {
                mean[i] += x[i] as f64 / n as f64;
            }
        }
        let r_emp = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
        let r_theo = vmf_mean_resultant(dim, kappa);
        assert!(
            (r_emp - r_theo).abs() < 0.05,
            "R_emp {r_emp} vs R_theo {r_theo}"
        );
    }

    #[test]
    fn fidelity_gap_is_below_one_for_clustered_data() {
        // Clustered real data must be farther from isotropy than samples of
        // its own vMF mixture (H2, experiment median ratio 0.016 in d=384).
        // Regime pinned by probe: d=64, 6 clusters, weak kappa ~ counts.
        let d = 64usize;
        let n_clusters = 6usize;
        let per = 100usize;
        let n = n_clusters * per;
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let noise = Normal::new(0.0_f32, 0.05).unwrap();
        let mut real = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            let c = i / per;
            for v in real.row_mut(i).iter_mut() {
                *v = rng.sample(noise);
            }
            real[[i, c * 2]] += 2.0;
            real[[i, c * 2 + 1]] += 1.0;
            let nn = real.row(i).dot(&real.row(i)).sqrt();
            real.row_mut(i).mapv_inplace(|v| v / nn);
        }
        let mut mus = Array2::<f32>::zeros((n_clusters, d));
        for c in 0..n_clusters {
            mus[[c, c * 2]] = 2.0;
            mus[[c, c * 2 + 1]] = 1.0;
            let nn = mus.row(c).dot(&mus.row(c)).sqrt();
            mus.row_mut(c).mapv_inplace(|v| v / nn);
        }
        let kappa: Vec<f64> = (0..n_clusters).map(|c| 10.0 - 1.0 * c as f64).collect();
        let cfg = EpConfig {
            n_directions: 32,
            ..EpConfig::default()
        };
        let report = fidelity_gap_ratio(real.view(), mus.view(), &kappa, &cfg);
        assert!(
            report.gap_ratio < 0.3,
            "ratio {} (real {} synth {})",
            report.gap_ratio,
            report.ep_real,
            report.ep_synth
        );
    }

    #[test]
    fn sampler_never_panics_across_kappas() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let dim = 16;
        let mut mu = Array1::<f32>::zeros(dim);
        mu[5] = 1.0;
        for &k in &[0.01, 0.5, 5.0, 50.0, 500.0] {
            let x = sample_vmf(&mut rng, mu.view(), k);
            let norm = x.dot(&x).sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "kappa {k}: norm {norm}");
        }
    }
}
