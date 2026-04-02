#!/usr/bin/env python3
"""Benchmark Python M2M - operations comparable to Rust criterion bench."""
import sys, time, json
import numpy as np

sys.path.insert(0, "C:/Users/Brian/Desktop/m2m-vector-search-main/src")
from m2m.config import M2MConfig
from m2m.splats import SplatStore
from m2m.clustering import KMeans
from m2m.encoding import FullEmbeddingBuilder
from m2m.energy import EnergyFunction
from m2m.geometry import normalize_sphere

def bench(label, fn, n_iter=10, n_warmup=3):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1e6  # us
        times.append(dt)
    avg = np.mean(times)
    med = np.median(times)
    print(f"  {label:50s} {avg:>12.1f} us  (median {med:.1f})")
    return {"name": label, "avg_us": float(avg), "median_us": float(med)}

def main():
    results = []
    print("=" * 75)
    print("M2M Python Benchmark")
    print("=" * 75)

    # 1. KMeans fit
    print("\n--- KMeans ---")
    for n in [100, 1000, 10000]:
        data = np.random.randn(n, 64).astype(np.float32)
        def make_fn(d=data):
            km = KMeans(n_clusters=10, max_iter=50, random_state=42)
            km.fit(d)
        iters = 3 if n >= 5000 else 5
        r = bench(f"kmeans_fit N={n:>6d} K=10 dim=64", make_fn, n_iter=iters, n_warmup=1)
        results.append(r)

    # 2. Encoding batch 1k (only batch works in Python)
    print("\n--- Encoding ---")
    builder = FullEmbeddingBuilder()
    n = 1000
    positions = np.random.randn(n, 3).astype(np.float32)
    colors = np.random.randn(n, 3).astype(np.float32)
    opacities = np.random.rand(n).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32)
    rotations = np.random.randn(n, 4).astype(np.float32)
    r = bench("full_embedding_batch_1k (640-d)", lambda: builder.build(positions, colors, opacities, scales, rotations))
    results.append(r)

    # 3. Energy
    print("\n--- Energy Functions ---")
    cfg = M2MConfig.simple()
    ef = EnergyFunction(cfg)
    batch = np.random.randn(100, 640).astype(np.float32)
    batch /= np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8
    r = bench("e_geom 100 vectors (640-d)", lambda: ef.E_geom(batch))
    results.append(r)

    # 4. Geometry
    print("\n--- Geometry ---")
    data = np.random.randn(10000, 640).astype(np.float32)
    r = bench("normalize_sphere 10k vectors (640-d)", lambda: normalize_sphere(data))
    results.append(r)

    # 5. KMeans transform
    print("\n--- Clustering Transform ---")
    km = KMeans(n_clusters=10, max_iter=50, random_state=42)
    train = np.random.randn(1000, 64).astype(np.float32)
    km.fit(train)
    query = np.random.randn(100, 64).astype(np.float32)
    r = bench("kmeans_transform 100 queries (dim=64)", lambda: km.transform(query))
    results.append(r)

    # 6. End-to-end find_neighbors
    print("\n--- End-to-end: find_neighbors ---")
    for n in [100, 500, 1000, 5000, 10000, 50000]:
        config = M2MConfig.simple()
        config.latent_dim = 640
        config.max_splats = n + 1000
        store = SplatStore(config)
        vecs = np.random.randn(n, 640).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        store.add_splat(vecs)
        store.build_index()
        q = np.random.randn(640).astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        for _ in range(3):
            store.find_neighbors(q, k=10)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            store.find_neighbors(q, k=10)
            times.append((time.perf_counter() - t0) * 1e6)
        avg = np.mean(times)
        med = np.median(times)
        print(f"  find_neighbors N={n:<6d} k=10              {avg:>12.1f} us  (median {med:.1f})")
        results.append({"name": f"find_neighbors N={n}", "avg_us": float(avg), "median_us": float(med)})

    with open("C:/Users/Brian/Desktop/m2m-rust/bench_python_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
