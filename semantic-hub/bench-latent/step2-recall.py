#!/usr/bin/env python3
"""Paso 2 del bench latente: recall del campo splat vs ground truth.

Por cada variante (splat-ingest, -hierarchical, -leader):
  - carga mu splats (default.bin) + landscape.bin (mu, alpha, kappa)
  - reconstruye membresía: vector crudo → centroide más cercano (L2)
  - ranking del campo: L2 puro vs energía gaussiana (alpha·exp(-kappa·d²))
  - métricas: contención top-10 (¿el GT está entre los miembros de los
    top-10 splats?), refine 2 etapas (top-3 splats → brute force interno →
    recall@10), y ms/query de cada etapa.
Baseline: HNSW crudo vía API 8871 (server del hub con los vectores completos).
"""
import glob
import json
import os
import struct
import sys
import time
import urllib.request
import numpy as np

HUB = "/mnt/d/splatdb/semantic-hub"
BL = f"{HUB}/bench-latent"
RAW = f"{HUB}/data/vectors/default.bin"


def load_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<Q", f.read(8))[0]
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()


def load_landscape(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<Q", f.read(8))[0]
        mu = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()
        rest = f.read()
    n_floats = len(rest) // 4
    vals = np.frombuffer(rest, dtype=np.float32)
    alpha, kappa = vals[:n], vals[n:2 * n]
    return mu, alpha.astype(np.float64), kappa.astype(np.float64)


def main():
    with open(f"{BL}/queries.json") as f:
        queries = json.load(f)
    Q = np.array([q["vec"] for q in queries], dtype=np.float32)
    GT = [q["gt"] for q in queries]
    X = load_bin(RAW)
    n_raw = X.shape[0]
    print(f"crudos: {n_raw} | queries: {len(Q)}")

    # ── baseline HNSW crudo (8871) ──
    recalls, lat = [], []
    for i, q in enumerate(queries):
        t0 = time.monotonic()
        req = urllib.request.Request(
            "http://127.0.0.1:8871/search",
            data=json.dumps({"query": q["q"], "embedding": q["vec"], "top_k": 10}).encode(),
            headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=60).read())["results"]
        lat.append((time.monotonic() - t0) * 1000)
        got = {r["index"] for r in res}
        recalls.append(len(got & set(GT[i])) / 10)
    print(f"\n[baseline HNSW crudo] recall@10={np.mean(recalls):.3f} "
          f"lat={np.mean(lat):.0f} ms/req (HTTP E2E)")

    # ── variantes splat ──
    for name in ["splat-ingest", "splat-ingest-hierarchical", "splat-ingest-leader"]:
        dd = f"/tmp/{name}"
        mu_path = f"{dd}/data/vectors/default.bin"
        land_path = f"{dd}/energy/landscape.bin"
        if not os.path.exists(mu_path):
            print(f"\n[{name}] SIN OUTPUT (ingest falló o no terminó)")
            continue
        MU = load_bin(mu_path).astype(np.float32)
        k = MU.shape[0]
        have_energy = os.path.exists(land_path)
        if have_energy:
            _, alpha, kappa = load_landscape(land_path)
        else:
            alpha = np.ones(k)
            kappa = np.ones(k)
        print(f"\n[{name}] {k} splats (ratio {n_raw / k:.1f}:1) | energy={'sí' if have_energy else 'no'}")

        # membresía por centroide más cercano
        d2_vz = np.sum((X ** 2), axis=1)[:, None]
        d2_mz = np.sum((MU ** 2), axis=1)[None, :]
        cross = X @ MU.T
        assign = np.argmin(d2_vz - 2 * cross + d2_mz, axis=1)
        members = [np.where(assign == j)[0] for j in range(k)]
        sizes = np.array([len(m) for m in members])
        print(f"  membresía: min={sizes.min()} mediana={int(np.median(sizes))} "
              f"max={sizes.max()} vacíos={(sizes == 0).sum()}")

        # scoring del campo (todas las queries a la vez)
        d2_qz = np.sum(Q ** 2, axis=1)[:, None]
        cross_q = Q @ MU.T
        D2 = d2_qz - 2 * cross_q + d2_mz  # (30, k) — d2_mz ya es (1, k)
        t0 = time.monotonic()
        top_l2 = np.argsort(D2, axis=1)[:, :10]
        t_l2 = (time.monotonic() - t0) / len(Q) * 1000
        t0 = time.monotonic()
        E = -D2 * kappa[None, :]
        E -= E.max(axis=1, keepdims=True)
        G = alpha[None, :] * np.exp(E)
        top_g = np.argsort(-G, axis=1)[:, :10]
        t_g = (time.monotonic() - t0) / len(Q) * 1000

        # orden de ranking L2 vs gaussiano: ¿coinciden?
        agree = np.mean([len(set(top_l2[i].tolist()) & set(top_g[i].tolist())) / 10
                         for i in range(len(Q))])

        for tag, top in [("L2", top_l2), ("gauss", top_g)]:
            cont = np.mean([len(set(GT[i]) & set(np.concatenate(
                [members[j] for j in top[i]] if sizes[top[i]].sum() else []))) / 10
                for i in range(len(Q))])
            # refine 2 etapas: top-3 splats → brute force entre miembros
            rec = []
            for i in range(len(Q)):
                cand = np.unique(np.concatenate(
                    [members[j] for j in top[i][:3]])) if sizes[top[i][:3]].sum() else np.array([])
                if len(cand) == 0:
                    rec.append(0.0)
                    continue
                sub = X[cand]
                d2 = np.sum((sub - Q[i]) ** 2, axis=1)
                local = np.argsort(d2)[:10]
                got = set(cand[local].tolist())
                rec.append(len(got & set(GT[i])) / 10)
            print(f"  [{tag}] contención(top10 splats)={cont:.3f} | "
                  f"recall@10 tras refine(top3)={np.mean(rec):.3f}")

        print(f"  timing campo: L2 {t_l2:.2f} ms/q | gauss {t_g:.2f} ms/q "
              f"(numpy, 30 queries) | overlap ranking L2-vs-gauss: {agree:.2f}")


if __name__ == "__main__":
    main()
