#!/usr/bin/env python3
"""Calidad del flat KMeans 10:1 (referencia numpy) contra GT.
Mismas métricas que step2: contención top-10 splats, refine top-3, timing campo.
"""
import json
import struct
import numpy as np

HUB = "/mnt/d/splatdb/semantic-hub"


def load_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<Q", f.read(8))[0]
        return np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d).copy()


with open(f"{HUB}/bench-latent/queries.json") as f:
    queries = json.load(f)
Q = np.array([q["vec"] for q in queries], dtype=np.float32)
GT = [q["gt"] for q in queries]
X = load_bin(f"{HUB}/data/vectors/default.bin")
C = np.load(f"{HUB}/bench-latent/flat-ref-centroids.npz")["mu"]
k = C.shape[0]
print(f"crudos={X.shape[0]} splats={k} (ratio {X.shape[0]/k:.1f}:1)")

# membresía
d2v = np.sum(X ** 2, axis=1)[:, None]
d2c = np.sum(C ** 2, axis=1)[None, :]
assign = np.argmin(d2v - 2 * X @ C.T + d2c, axis=1)
members = [np.where(assign == j)[0] for j in range(k)]
sizes = np.array([len(m) for m in members])

# scoring campo
D2 = np.sum(Q ** 2, axis=1)[:, None] - 2 * Q @ C.T + d2c
top_l2 = np.argsort(D2, axis=1)[:, :10]

# alpha/kappa como kmeans_to_splats: alpha=n_members/n, kappa=clamp(1/var, .1, 100)
alpha = sizes / X.shape[0]
kappas = np.empty(k)
for j in range(k):
    if sizes[j] > 0:
        mu = C[j]
        var = np.mean(np.sum((X[members[j]] - mu) ** 2, axis=1))
        kappas[j] = np.clip(1.0 / max(var, 1e-9), 0.1, 100.0)
    else:
        kappas[j] = 1.0
E = -D2 * kappas[None, :]
E -= E.max(axis=1, keepdims=True)
G = alpha[None, :] * np.exp(E)
top_g = np.argsort(-G, axis=1)[:, :10]
agree = np.mean([len(set(top_l2[i]) & set(top_g[i])) / 10 for i in range(len(Q))])

for tag, top in [("L2", top_l2), ("gauss", top_g)]:
    cont = np.mean([len(set(GT[i]) & set(np.concatenate(
        [members[j] for j in top[i]]))) / 10 for i in range(len(Q))])
    rec = []
    for i in range(len(Q)):
        cand = np.unique(np.concatenate([members[j] for j in top[i][:3]]))
        sub = X[cand]
        d2 = np.sum((sub - Q[i]) ** 2, axis=1)
        local = np.argsort(d2)[:10]
        rec.append(len(set(cand[local].tolist()) & set(GT[i])) / 10)
    print(f"[{tag}] contención(top10)={cont:.3f} | recall@10 refine(top3)={np.mean(rec):.3f}")
print(f"overlap ranking L2 vs gauss: {agree:.2f}")
