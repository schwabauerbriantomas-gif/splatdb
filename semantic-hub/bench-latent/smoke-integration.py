#!/usr/bin/env python3
"""Smoke E2E: hub 8871 /search GPU + field 8872 /field-search y /search."""
import json
import time
import urllib.request


def post(port, path, payload):
    t0 = time.monotonic()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=30)
    return json.loads(r.read()), (time.monotonic() - t0) * 1000


q = json.load(open("/mnt/d/splatdb/semantic-hub/bench-latent/queries.json"))[0]
emb = q["vec"]
gt = set(q["gt"])

r, ms = post(8871, "/search", {"query": "q", "embedding": emb, "top_k": 10})
got = {x["index"] for x in r["results"]}
print(f"hub 8871 /search: backend={r.get('backend')} lat={ms:.0f}ms recall={len(got & gt)/10:.1f}")

r2, ms2 = post(8872, "/field-search", {"embedding": emb, "top_k": 10})
got2 = {x["index"] for x in r2["results"]}
print(f"field 8872 /field-search: lat={ms2:.0f}ms recall={len(got2 & gt)/10:.1f} "
      f"visitó {r2['clusters_visited']}/{r2['n_splats']} clusters, scan {r2['scan_frac']*100:.0f}%")

r3, ms3 = post(8872, "/search", {"query": "q", "embedding": emb, "top_k": 10})
print(f"field 8872 /search (GPU sobre splats): backend={r3.get('backend')} lat={ms3:.0f}ms")
