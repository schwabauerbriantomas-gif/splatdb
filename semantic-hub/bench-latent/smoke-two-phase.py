#!/usr/bin/env python3
"""Smoke E2E: /field-search two-phase vs exacto en el hub 8871."""
import json, urllib.request

qs = json.load(open("semantic-hub/bench-latent/queries.json"))
emb = (qs[0].get("embedding") or qs[0].get("vec"))[:768]

def post(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8872/field-search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

exact = post({"embedding": emb, "top_k": 10})
two = post({"embedding": emb, "top_k": 10, "mode": "two-phase", "nprobe": 32})
two1 = post({"embedding": emb, "top_k": 10, "mode": "two-phase", "nprobe": 1})

def ids(r):
    return [x["index"] for x in r["results"]]

print("exacto backend:", exact["backend"], "| time_ms:", exact["time_ms"])
print("two-phase backend:", two["backend"], "| nprobe:", two["nprobe"], "| time_ms:", two["time_ms"])
print("q0 top-10 exacto:", ids(exact))
print("q0 top-10 two32 :", ids(two))
inter = len(set(ids(exact)) & set(ids(two)))
print(f"recall@10 two-phase(nprobe=32) vs exacto: {inter}/10")
print(f"two-phase nprobe=1: scan_frac={two1['scan_frac']}, recall={len(set(ids(exact)) & set(ids(two1)))}/10")
print("fases two32: phase1_us:", two["phase1_us"], "| phase2_us:", two["phase2_us"], "| scanned:", two["vectors_scanned"], f"({two['scan_frac']*100:.1f}%)")
