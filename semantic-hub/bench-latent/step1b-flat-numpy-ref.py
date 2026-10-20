#!/usr/bin/env python3
"""Referencia de calidad del flat KMeans (Lloyd's) en numpy — mismo algoritmo
que simple_kmeans de Rust (init LCG sobre filas aleatorias, k=2627, 20 iters,
f64 en update) pero vectorizado. Representa la calidad "antes/después" de la
optimización Rust: la implementación optimizada debe matchear este recall.
Deja memberships en bench-latent/flat-ref.npz (labels + tiempo).
"""
import json
import struct
import time
import numpy as np

HUB = "/mnt/d/splatdb/semantic-hub"
K = 2627
MAX_ITER = 20
SEED = 42


def lcg_rows(n, k, seed):
    state = seed
    idx = []
    for _ in range(k):
        state = (state * 6364136223846793005 + 1) % (1 << 64)
        idx.append(state % n)
    return np.array(idx, dtype=np.int64)


def main():
    with open(f"{HUB}/data/vectors/default.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<Q", f.read(8))[0]
        X = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)

    C = X[lcg_rows(n, K, SEED)].astype(np.float32).copy()
    labels = np.zeros(n, dtype=np.int64)
    x2 = np.sum(X.astype(np.float32) ** 2, axis=1)[:, None]
    t0 = time.monotonic()
    for it in range(MAX_ITER):
        # d² = |x|² - 2 x·c + |c|²  → argmin
        d2 = x2 - 2.0 * (X @ C.T) + np.sum(C ** 2, axis=1)[None, :]
        new = np.argmin(d2, axis=1)
        changed = int((new != labels).sum())
        labels = new
        # update f64
        sums = np.zeros((K, d), dtype=np.float64)
        np.add.at(sums, labels, X.astype(np.float64))
        counts = np.bincount(labels, minlength=K).astype(np.float64)
        nz = counts > 0
        C[nz] = (sums[nz] / counts[nz, None]).astype(np.float32)
        dt = time.monotonic() - t0
        print(f"iter {it}: changed={changed} ({dt:.0f}s acumulado)", flush=True)
        if changed == 0:
            break
    total = time.monotonic() - t0
    counts = np.bincount(labels, minlength=K)
    print(f"LISTO: {total:.1f}s | clusters vacíos: {(counts==0).sum()} | "
          f"tamaño mediana: {int(np.median(counts[counts>0]))}")
    np.savez(f"{HUB}/bench-latent/flat-ref.npz", labels=labels, seconds=total, k=K)
    # centroids finales también, para el step2
    np.savez(f"{HUB}/bench-latent/flat-ref-centroids.npz", mu=C)
    print(f"OK → bench-latent/flat-ref.npz")


if __name__ == "__main__":
    main()
