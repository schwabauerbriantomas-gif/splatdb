//! Field search — el campo splat usado con geometría correcta.
//!
//! El concepto de SplatsDB (campo de gaussianas) solo funciona como índice si
//! cada splat se trata como una CELDA CON RADIO y la búsqueda usa COTAS:
//!   d(q, x) ≥ d(q, mu_c) − R_c   (desigualdad triangular, R_c = radio del cluster)
//! Sweep ordenado por d(q, mu) escaneando todo cluster con cota < k-ésimo mejor
//! ⇒ resultado EXACTO (recall 1.0 salvo empates flotantes) saltando clusters.
//!
//! El scoring de energía alpha·exp(−kappa·d²) NO es un bound y en 768d destruye
//! el ranking (alpha domina = popularidad) — queda solo para diversidad/temas.
//!
//! Layout en disco (aditivo, no rompe nada existente):
//!   data/vectors/default.bin  : splats (mu)          — ya lo escribía el ingest
//!   data/vectors/raw.bin      : vectores crudos      — nuevo (miembros)
//!   field/members.idx         : u64 n, u64 k, u64 offsets[k+1], u32 ids[n]
//!   field/radii.bin           : u64 k, f32 radii[k]
//!
//! Envelope honesto: la poda es efectiva cuando los radios son chicos respecto
//! de las distancias query-centroide (n grande, o dims bajas vía MRL). A n=26k
//! ×768d poda ~5-10% — el valor aparece a escala (≥500k) o como capa exacta.

use ndarray::{Array1, Array2, Axis};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Asigna cada vector crudo al splat más cercano (gemm), calcula radios y
/// persiste: shard "raw" (crudos) + field/members.idx + field/radii.bin.
/// Llamar después de guardar los splats en "default".
pub fn persist_field(
    data_dir: &str,
    backend: &str,
    raw: &Array2<f32>,
    mu: &Array2<f32>,
) -> Result<(usize, f64), String> {
    let n = raw.nrows();
    let k = mu.nrows();
    if k == 0 || n == 0 {
        return Err("field: store vacío".to_string());
    }

    let dots = raw.dot(&mu.t());
    let x2: Array1<f32> = raw.map_axis(Axis(1), |r| r.dot(&r));
    let m2: Array1<f32> = mu.map_axis(Axis(1), |r| r.dot(&r));
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let row = dots.row(i);
        let mut best = 0usize;
        let mut bv = f32::INFINITY;
        for j in 0..k {
            let v = x2[i] - 2.0 * row[j] + m2[j];
            if v < bv {
                bv = v;
                best = j;
            }
        }
        labels[i] = best;
    }

    let mut counts = vec![0usize; k];
    for &l in &labels {
        counts[l] += 1;
    }
    let mut rmax = vec![0.0f32; k];
    for i in 0..n {
        let l = labels[i];
        let d2 = (x2[i] - 2.0 * dots[[i, l]] + m2[l]).max(0.0);
        let d = d2.sqrt();
        if d > rmax[l] {
            rmax[l] = d;
        }
    }

    let mut offsets = vec![0u64; k + 1];
    for c in 0..k {
        offsets[c + 1] = offsets[c] + counts[c] as u64;
    }
    let mut cursor = offsets[..k].to_vec();
    let mut ids = vec![0u32; n];
    for (i, &l) in labels.iter().enumerate() {
        ids[cursor[l] as usize] = i as u32;
        cursor[l] += 1;
    }

    // crudos como shard "raw" (reusa la persistence existente)
    let persist =
        super::helpers::make_persistence(data_dir, backend, true).map_err(|e| e.to_string())?;
    persist
        .save_vectors(raw, "raw")
        .map_err(|e| e.to_string())?;

    let dir = Path::new(data_dir).join("field");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    // Atomic writes: a crash mid-ingest must not leave a truncated
    // members.idx/radii.bin that load_field would reject (or worse, trust).
    {
        let mut buf = Vec::with_capacity(16 + (offsets.len() + ids.len()) * 8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        buf.extend_from_slice(&(k as u64).to_le_bytes());
        for o in &offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        for id in &ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        splatsdb::storage::persistence::atomic_write(&dir.join("members.idx"), &buf)
            .map_err(|e| e.to_string())?;
    }
    {
        let mut buf = Vec::with_capacity(8 + rmax.len() * 4);
        buf.extend_from_slice(&(k as u64).to_le_bytes());
        for r in &rmax {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        splatsdb::storage::persistence::atomic_write(&dir.join("radii.bin"), &buf)
            .map_err(|e| e.to_string())?;
    }

    let empty = counts.iter().filter(|&&c| c == 0).count();
    let mut sorted_r = rmax.clone();
    sorted_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = if sorted_r.is_empty() {
        0.0
    } else {
        sorted_r[sorted_r.len() / 2]
    };
    eprintln!(
        "[field] persistido: {} splats (ratio {:.2}:1), {} vacíos, radio mediana {:.3}",
        k,
        n as f64 / k as f64,
        empty,
        med
    );
    Ok((k, n as f64 / k as f64))
}

fn load_u64s(path: &Path, count: usize, skip: usize) -> Result<Vec<u64>, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start((skip * 8) as u64))
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; count * 8];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect())
}

fn load_u32s(path: &Path, count: usize, skip_bytes: u64) -> Result<Vec<u32>, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(skip_bytes))
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect())
}

fn load_f32s(path: &Path, count: usize, skip: usize) -> Result<Vec<f32>, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start((skip * 4) as u64))
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// Búsqueda two-phase (estilo IVF). Fase 1: top-`nprobe` clusters por d(q, mu)
/// — kernel PTX si use_gpu y hay driver, sino ndarray. Fase 2: re-rank exacto
/// L2² (rayon) de los miembros de esos clusters vía CSR.
///
/// APROXIMADO: el recall depende de nprobe (medir contra field_search, exacto).
/// A 26k×768d el exacto directo gana; two-phase paga cuando K crece y los
/// crudos no caben en memoria rápida (a escala, raw.bin se streamería por
/// clusters en vez de cargarse entero).
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub fn field_search_two_phase(
    data_dir: &str,
    q: &[f32],
    kq: usize,
    nprobe: usize,
    use_gpu: bool,
) -> Result<serde_json::Value, String> {
    if q.len() < 8 {
        return Err("query demasiado corta".to_string());
    }
    let t0 = std::time::Instant::now();
    let base = Path::new(data_dir);
    let persist =
        super::helpers::make_persistence(data_dir, "sqlite", false).map_err(|e| e.to_string())?;

    let raw = persist
        .load_vectors("raw")
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "no hay shard raw (correr ingest con field)".to_string())?;
    let splats = persist
        .load_vectors("default")
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "no hay shard default (splats)".to_string())?;

    // Validar dimensión ANTES de map_axis/dot: q más corta o más larga que el
    // campo paniquea (subtract shape mismatch) en vez de devolver error limpio.
    if q.len() != splats.ncols() || q.len() != raw.ncols() {
        return Err(format!(
            "dim mismatch: query={} splats={} raw={}",
            q.len(),
            splats.ncols(),
            raw.ncols()
        ));
    }

    let mpath = base.join("field").join("members.idx");
    let hdr = load_u64s(&mpath, 2, 0)?;
    let (n, k) = (hdr[0] as usize, hdr[1] as usize);
    if n != raw.nrows() || k != splats.nrows() {
        return Err(format!(
            "field desincronizado: members n={n} k={k} vs raw={} splats={}",
            raw.nrows(),
            splats.nrows()
        ));
    }
    let offsets_vec = load_u64s(&mpath, k + 1, 2)?;
    let ids = load_u32s(&mpath, n, ((2 + k + 1) * 8) as u64)?;

    // ── Fase 1: top-nprobe clusters por d(q, mu)
    let nprobe = nprobe.min(k).max(1);
    let t1 = std::time::Instant::now();
    let mut winners: Vec<usize> = Vec::with_capacity(nprobe);

    #[cfg(feature = "cuda")]
    let mut gpu_used = false;
    #[cfg(not(feature = "cuda"))]
    let gpu_used = false;

    #[cfg(feature = "cuda")]
    if use_gpu {
        if let Some(mut gpu) = splatsdb::gpu::cuda_kernel::GpuIndex::new() {
            let flat: Vec<f32> = splats.iter().copied().collect();
            if gpu.upload_dataset(&flat, k, splats.ncols()) {
                if let Some(res) = gpu.topk_search(q, 1, nprobe, "l2") {
                    winners = res[0].0.clone();
                    gpu_used = true;
                }
            }
        }
    }
    if !gpu_used {
        let qv = ndarray::arr1(q);
        let d_mu: Array1<f32> = splats.map_axis(Axis(1), |r| {
            let diff = r.to_owned() - &qv;
            diff.dot(&diff).max(0.0).sqrt()
        });
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|a, b| {
            d_mu[*a]
                .partial_cmp(&d_mu[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        winners = order.into_iter().take(nprobe).collect();
    }
    let phase1_us = t1.elapsed().as_micros() as u64;

    // ── Fase 2: gather de miembros + re-rank exacto L2² (rayon)
    let t2 = std::time::Instant::now();
    let mut cand: Vec<u32> = Vec::new();
    let mut scanned = 0usize;
    for &c in &winners {
        let lo = offsets_vec[c] as usize;
        let hi = offsets_vec[c + 1] as usize;
        scanned += hi - lo;
        cand.extend_from_slice(&ids[lo..hi]);
    }
    use rayon::prelude::*;
    let mut top: Vec<(f32, u32)> = cand
        .par_iter()
        .map(|&id| {
            let row = raw.row(id as usize);
            let diff = row.to_owned() - ndarray::arr1(q);
            (diff.dot(&diff), id)
        })
        .collect();
    top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(kq);
    let phase2_us = t2.elapsed().as_micros() as u64;

    let results: Vec<serde_json::Value> = top
        .iter()
        .map(|(d, id)| serde_json::json!({ "id": *id as u64, "dist2": *d }))
        .collect();
    let pct = |x: f64| (x * 10000.0).round() / 10000.0;
    Ok(serde_json::json!({
        "method": "field_two_phase_ivf",
        "k": kq,
        "n_vectors": n,
        "n_splats": k,
        "nprobe": nprobe,
        "gpu_phase1": gpu_used,
        "clusters_probed": winners.len(),
        "vectors_scanned": scanned,
        "scan_frac": pct(scanned as f64 / n as f64),
        "phase1_us": phase1_us,
        "phase2_us": phase2_us,
        "time_ms": t0.elapsed().as_millis() as u64,
        "results": results,
    }))
}

/// Búsqueda exacta por cotas. q = query float; resultado JSON con métricas.
pub fn field_search(data_dir: &str, q: &[f32], kq: usize) -> Result<serde_json::Value, String> {
    if q.len() < 8 {
        return Err("query demasiado corta".to_string());
    }
    let t0 = std::time::Instant::now();
    let base = Path::new(data_dir);
    let persist =
        super::helpers::make_persistence(data_dir, "sqlite", false).map_err(|e| e.to_string())?;

    let raw = persist
        .load_vectors("raw")
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "no hay shard raw (correr ingest con field)".to_string())?;
    let splats = persist
        .load_vectors("default")
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "no hay shard default (splats)".to_string())?;

    // Validar dimensión ANTES de map_axis: q != ncols() paniquea en el
    // subtract (shape mismatch) en vez de devolver error limpio.
    if q.len() != splats.ncols() || q.len() != raw.ncols() {
        return Err(format!(
            "dim mismatch: query={} splats={} raw={}",
            q.len(),
            splats.ncols(),
            raw.ncols()
        ));
    }

    let mpath = base.join("field").join("members.idx");
    let hdr = load_u64s(&mpath, 2, 0)?;
    let (n, k) = (hdr[0] as usize, hdr[1] as usize);
    if n != raw.nrows() || k != splats.nrows() {
        return Err(format!(
            "field desincronizado: members n={n} k={k} vs raw={} splats={}",
            raw.nrows(),
            splats.nrows()
        ));
    }
    let offsets_vec = load_u64s(&mpath, k + 1, 2)?;
    let ids = load_u32s(&mpath, n, ((2 + k + 1) * 8) as u64)?;
    let radii = load_f32s(&base.join("field").join("radii.bin"), k, 1)?;

    let qv = ndarray::arr1(q);
    let d_mu: Array1<f32> = splats.map_axis(Axis(1), |r| {
        let diff = r.to_owned() - &qv;
        diff.dot(&diff).max(0.0).sqrt()
    });
    let lb: Array1<f32> = Array1::from_iter((0..k).map(|c| (d_mu[c] - radii[c]).max(0.0)));

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| {
        d_mu[*a]
            .partial_cmp(&d_mu[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top: Vec<(f32, u32)> = Vec::with_capacity(kq + 8);
    let mut kth = f32::INFINITY;
    let mut visited = 0usize;
    let mut scanned = 0usize;

    for &c in &order {
        let lb_c = lb[c] * lb[c];
        if !(lb_c < kth || top.len() < kq) {
            continue; // cota no monotónica en d_mu: sin break, pero el skip es exacto
        }
        let lo = offsets_vec[c] as usize;
        let hi = offsets_vec[c + 1] as usize;
        if hi == lo {
            continue;
        }
        visited += 1;
        scanned += hi - lo;
        for &id in &ids[lo..hi] {
            let row = raw.row(id as usize);
            let diff = row.to_owned() - &qv;
            let d2 = diff.dot(&diff);
            if top.len() < kq || d2 < kth {
                top.push((d2, id));
                top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                if top.len() > kq {
                    top.truncate(kq);
                }
                kth = if top.len() >= kq {
                    top[top.len() - 1].0
                } else {
                    f32::INFINITY
                };
            }
        }
    }

    let results: Vec<serde_json::Value> = top
        .iter()
        .map(|(d, id)| serde_json::json!({ "id": *id as u64, "dist2": *d }))
        .collect();
    let pct = |x: f64| (x * 10000.0).round() / 10000.0;
    Ok(serde_json::json!({
        "method": "field_exact_bounds",
        "k": kq,
        "n_vectors": n,
        "n_splats": k,
        "clusters_visited": visited,
        "cluster_frac": pct(visited as f64 / k as f64),
        "vectors_scanned": scanned,
        "scan_frac": pct(scanned as f64 / n as f64),
        "time_ms": t0.elapsed().as_millis() as u64,
        "results": results,
    }))
}
