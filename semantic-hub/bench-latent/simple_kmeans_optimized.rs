// KMeans optimizado para SplatsDB — reemplazo de simple_kmeans (dataset_transformer.rs:835)
//
// Cambios vs original:
// 1. Asignación por gemm (data.dot(centroids.t())) en vez de doble loop escalar f64
//    — O(n·k·d) en BLAS multithread en vez de un core.
// 2. Update con acumuladores f64 agrupados por label (misma precisión que el original,
//    sin el escaneo double-loop sobre toda la matriz).
// 3. Mismo init LCG + mismas filas aleatorias → mismo comportamiento esperado;
//    pequeñas diferencias f32-vs-f64 en distancias pueden mover labels en empates.
// 4. Early-exit por convergencia igual que el original.

fn simple_kmeans(
    data: &Array2<f32>,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let (n, dim) = data.dim();
    let k = k.min(n);
    if k == 0 || dim == 0 {
        return (Vec::new(), vec![0; n]);
    }

    // init idéntico al original: LCG sobre filas aleatorias
    let mut rng_state = seed;
    let mut centroids = Array2::<f32>::zeros((k, dim));
    for mut row in centroids.rows_mut() {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let idx = (rng_state as usize) % n;
        row.assign(&data.row(idx));
    }

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // ── asignación: d²(i,c) = |x|² − 2·x·c + |c|²  ⇒  argmin_c (|c|²/2 − x·c)
        let dots = data.dot(&centroids.t()); // (n, k) gemm
        let mut best_c = vec![0usize; n];
        let mut best_v = vec![f32::NEG_INFINITY; n];
        for c in 0..k {
            let half_norm: f32 = centroids.row(c).dot(&centroids.row(c)) * 0.5;
            for (i, &dot) in dots.column(c).iter().enumerate() {
                let score = dot - half_norm;
                if score > best_v[i] {
                    best_v[i] = score;
                    best_c[i] = c;
                }
            }
        }
        let mut changed = false;
        for i in 0..n {
            if labels[i] != best_c[i] {
                labels[i] = best_c[i];
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // ── update: acumular f64 por label en una pasada
        let mut counts = vec![0usize; k];
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; dim]; k];
        for (i, &c) in labels.iter().enumerate() {
            counts[c] += 1;
            let s = &mut sums[c];
            let row = data.row(i);
            for j in 0..dim {
                s[j] += row[j] as f64;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for j in 0..dim {
                    centroids[[c, j]] = (sums[c][j] * inv) as f32;
                }
            }
        }
    }
    (centroids.rows().into_iter().map(|r| r.to_vec()).collect(), labels)
}
