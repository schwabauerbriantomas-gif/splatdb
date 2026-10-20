#!/usr/bin/env python3
"""Bench two-phase IVF vs exacto por cotas — 30 queries del GT, dataset hub 26k."""
import json, subprocess, statistics, sys

BIN = "./target/release/splatsdb"
DATA = "/tmp/splat-field"
QUERIES = "semantic-hub/bench-latent/queries.json"
K = 10

qs = json.load(open(QUERIES))
vecs = []
for q in qs:
    emb = (q.get("embedding") or q.get("vec")) if isinstance(q, dict) else q
    vecs.append(emb)

def run(mode_args, vec):
    cmd = [BIN, "--dim", "768", "--data-dir", DATA, "field-search",
           f"--query={','.join(f'{x:.6f}' for x in vec[:768])}", f"--k={K}"] + mode_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)

# GT = exacto por cotas (recall 0.99 vs brute-force; misma referencia)
gt_ids = []
for v in vecs:
    out = run([], v)
    gt_ids.append([r["id"] for r in out["results"]])
print(f"GT exacto: 30 queries, top-{K}", file=sys.stderr)

def sweep(nprobe, gpu):
    recalls, p1s, p2s, scans, walls = [], [], [], [], []
    gpu_hits = 0
    for i, v in enumerate(vecs):
        out = run((["--nprobe", str(nprobe)] + (["--gpu"] if gpu else [])), v)
        if out is None:
            continue
        ids = [r["id"] for r in out["results"]]
        recalls.append(len(set(ids) & set(gt_ids[i])) / K)
        p1s.append(out["phase1_us"])
        p2s.append(out["phase2_us"])
        scans.append(out["scan_frac"])
        walls.append(out["time_ms"])
        gpu_hits += 1 if out.get("gpu_phase1") else 0
    n = len(recalls)
    print(json.dumps({
        "mode": f"nprobe={nprobe}" + ("+gpu" if gpu else "-cpu"),
        "n": n,
        "recall_mean": round(statistics.mean(recalls), 4),
        "recall_min": round(min(recalls), 2),
        "scan_frac_mean": round(statistics.mean(scans), 4),
        "phase1_us_mean": round(statistics.mean(p1s)),
        "phase2_us_mean": round(statistics.mean(p2s)),
        "wall_ms_mean": round(statistics.mean(walls), 1),
        "gpu_phase1_hits": gpu_hits,
    }))

for np_ in [1, 4, 16, 32, 64, 128]:
    sweep(np_, False)
for np_ in [32, 128]:
    sweep(np_, True)
