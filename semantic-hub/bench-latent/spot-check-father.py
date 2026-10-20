#!/usr/bin/env python3
"""Spot-check del padre: resuelve contradicción V3 vs V1 sobre el campo backend."""
import json
import time
import urllib.request


def post(path, payload, port=8871):
    t0 = time.monotonic()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=30)
    return json.loads(r.read()), (time.monotonic() - t0) * 1000


qs = json.load(open("/mnt/d/splatdb/semantic-hub/bench-latent/queries.json"))
for i in (0, 14, 29):
    r, ms = post("/search", {"query": "q", "embedding": qs[i]["vec"], "top_k": 10})
    got = {x["index"] for x in r["results"]}
    print(f"q{i}: backend={r.get('backend')!r} "
          f"recall={len(got & set(qs[i]['gt']))/10:.1f} lat={ms:.1f}ms")

r, ms = post("/search", {"query": "scraper licitaciones Argentina", "top_k": 3})
print(f"texto: backend={r.get('backend')!r} results={len(r['results'])} lat={ms:.0f}ms")
