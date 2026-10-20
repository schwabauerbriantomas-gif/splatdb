#!/usr/bin/env python3
"""Prototipo del concepto arreglado: splat = celda con radio, búsqueda por cotas.

Búsqueda EXACTA (recall 1.0 por construcción) con poda de clusters:
  - splats ordenados por d(q, mu)
  - lower bound de un cluster no visitado: (d(q,mu) − R)²  [R = radio del cluster]
  - se corta cuando el bound supera el k-ésimo mejor encontrado
Métricas: recall (==1.0 esperado), % clusters visitados, % vectores escaneados,
ms/query vs brute force (53 ms a esta escala).
"""
import json
import struct
import time
import numpy as np

HUB = "/mnt/d/splatdb/semantic-hub"
K = 2627


def load_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<Q", f.read(8))[0]
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()


def build_field(X, k, seed=42):
    """KMeans flat (Lloyd, mismo que simple_kmeans) + radios por cluster."""
    rng = np.random.default_rng(0)
    state = seed
    idx = []
    for _ in range(k):
        state = (state * 6364136223846793005 + 1) % (1 << 64)
        idx.append(state % X.shape[0])
    C = X[np.array(idx)].copy()
    labels = np.zeros(X.shape[0], dtype=np.int64)
    x2 = np.sum(X ** 2, axis=1)
    for it in range(20):
        d2 = x2[:, None] - 2 * (X @ C.T) + np.sum(C ** 2, axis=1)[None, :]
        new = np.argmin(d2, axis=1)
        if (new == labels).all():
            break
        labels = new
        sums = np.zeros((C.shape[0], X.shape[1]), dtype=np.float64)
        np.add.at(sums, labels, X.astype(np.float64))
        counts = np.bincount(labels, minlength=C.shape[0])
        nz = counts > 0
        C[nz] = (sums[nz] / counts[nz, None]).astype(np.float32)
    # radios: distancia máxima de cada miembro a su centroide (bound seguro)
    d2c = np.sum((X - C[labels]) ** 2, axis=1)
    radii = np.sqrt(np.maximum(np.bincount(labels, weights=np.sqrt(d2c), minlength=C.shape[0]) * 0, 0))
    rmax = np.zeros(C.shape[0])
    np.maximum.at(rmax, labels, np.sqrt(d2c))
    return C, labels, rmax


def field_search(q, X, C, offsets, radii, kq=10):
    """Exact search con poda por cotas. Devuelve (top-k, clusters_visitados, vecs_escaneados)."""
    d_mu = np.sqrt(np.maximum(np.sum((C - q) ** 2, axis=1), 0))
    order = np.argsort(d_mu)
    lb = np.maximum(d_mu - radii, 0.0)  # lower bound por cluster

    kth = np.inf
    top = []
    visited = 0
    scanned = 0
    # sweep exacto: orden por d_mu (los clusters con puntos cercanos van primero
    # y ajustan kth temprano); se escanea todo cluster con lb² < kth. Un cluster
    # salteado tiene lb ≥ kth_final ⇒ ninguno de sus puntos entra al top-k.
    for c in order:
        lb_c = lb[c] * lb[c]
        if lb_c < kth or visited < kq:
            lo, hi = offsets[c], offsets[c + 1]
            if hi == lo:
                continue
            members = _members[lo:hi]
            sub = X[members]
            d2 = np.sum((sub - q) ** 2, axis=1)
            scanned += len(members)
            visited += 1
            for j, d in enumerate(d2):
                if d < kth:
                    top.append((d, int(members[j])))
                    top.sort()
                    if len(top) > kq:
                        top = top[:kq]
                    # kth solo poda cuando el top está COMPLETO: con top parcial
                    # el peor de N<k puntos no es una cota válida del k-ésimo real.
                    kth = top[-1][0] if len(top) >= kq else np.inf
    ids = [t[1] for t in top[:kq]]
    return ids, visited, scanned


# members global para no re-pasar
_members = None


def main():
    global _members
    X = load_bin(f"{HUB}/data/vectors/default.bin")
    n = X.shape[0]
    with open(f"{HUB}/bench-latent/queries.json") as f:
        queries = json.load(f)
    Q = np.array([q["vec"] for q in queries], dtype=np.float32)
    GT = [q["gt"] for q in queries]

    print(f"construyendo campo {K}:{n} (kmeans + radios)...")
    C, labels, radii = build_field(X, K)
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=C.shape[0])
    offsets = np.zeros(C.shape[0] + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    _members = order.astype(np.int64)
    print(f"campo: {C.shape[0]} splats, radios: mediana={np.median(radii[radii>0]):.3f} "
          f"p90={np.percentile(radii[radii>0], 90):.3f}")

    t0 = time.monotonic()
    recalls, vfrac, sfrac, times = [], [], [], []
    for i, q in enumerate(Q):
        t1 = time.monotonic()
        ids, v, s = field_search(q, X, C, offsets, radii)
        times.append(time.monotonic() - t1)
        recalls.append(len(set(ids) & set(GT[i])) / 10)
        vfrac.append(v / C.shape[0])
        sfrac.append(s / n)
    print(f"\n[FIELD EXACT {K}:1] recall@10={np.mean(recalls):.3f} (debe ser 1.000)")
    print(f"  clusters visitados: {np.mean(vfrac)*100:.1f}% | vectores escaneados: {np.mean(sfrac)*100:.1f}%")
    print(f"  latencia: media={np.mean(times)*1000:.1f} ms p95={np.percentile(times,95)*1000:.1f} ms "
          f"(brute force era 53 ms/query)")


if __name__ == "__main__":
    main()
