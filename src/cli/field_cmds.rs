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
use std::io::{Read, Seek, SeekFrom, Write};
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

    let mut f = fs::File::create(dir.join("members.idx")).map_err(|e| e.to_string())?;
    f.write_all(&(n as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    f.write_all(&(k as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for o in &offsets {
        f.write_all(&o.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    for id in &ids {
        f.write_all(&id.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    drop(f);
    let mut f = fs::File::create(dir.join("radii.bin")).map_err(|e| e.to_string())?;
    f.write_all(&(k as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for r in &rmax {
        f.write_all(&r.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    drop(f);

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
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn load_u32s(path: &Path, count: usize, skip_bytes: u64) -> Result<Vec<u32>, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(skip_bytes))
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn load_f32s(path: &Path, count: usize, skip: usize) -> Result<Vec<f32>, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start((skip * 4) as u64))
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
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
