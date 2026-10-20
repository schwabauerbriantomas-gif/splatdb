#!/usr/bin/env python3
"""Búsqueda semántica sobre el hub (SplatsDB 8871 + sidecar).

Uso:
  sem-search.py "query" [--k 8] [--repo nombre] [--kind repo|memory] [--full]
"""
import argparse
import json
import sqlite3
import urllib.request

EMB = "http://127.0.0.1:8860/v1/embeddings"
SPLAT = "http://127.0.0.1:8871"
HUB = "/mnt/d/splatdb/semantic-hub"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=8, help="resultados a mostrar")
    ap.add_argument("--fetch", type=int, default=300,
                    help="candidatos a traer antes de filtrar (memoria es ~6%% del índice: no bajar de 200)")
    ap.add_argument("--repo", help="filtrar por repo")
    ap.add_argument("--kind", help="filtrar por kind (repo|memory)")
    ap.add_argument("--full", action="store_true", help="mostrar texto completo")
    args = ap.parse_args()

    req = urllib.request.Request(EMB, data=json.dumps({"input": [args.query]}).encode(),
                                 headers={"Content-Type": "application/json"})
    vec = json.loads(urllib.request.urlopen(req, timeout=60).read())["data"][0]["embedding"]

    req = urllib.request.Request(SPLAT + "/search",
                                 data=json.dumps({"query": args.query, "embedding": vec,
                                                  "top_k": args.fetch}).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read())["results"]
    ranked = sorted(res, key=lambda r: r["score"])

    con = sqlite3.connect(f"{HUB}/sidecar.db")
    shown = 0
    for r in ranked:
        row = con.execute("SELECT repo, path, kind, meta, text FROM chunks WHERE rowid=?",
                          (r["index"] + 1,)).fetchone()
        if row is None:
            continue
        repo, path, kind, meta, text = row
        if args.repo and repo != args.repo:
            continue
        if args.kind and kind != args.kind:
            continue
        shown += 1
        dist = r["score"]
        print(f"#{shown} [{repo} :: {path}] kind={kind} score={dist:.3f} (menor=mejor)")
        snippet = text if args.full else text[:220].replace("\n", " ")
        print(f"    {snippet}")
        if shown >= args.k:
            break
    if shown == 0:
        print("sin resultados (¿hub levantado? ¿merge corrido?)")


if __name__ == "__main__":
    main()
