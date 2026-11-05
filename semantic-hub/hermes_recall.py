#!/usr/bin/env python3
"""recall — búsqueda semántica sobre sesiones Hermes (hub :8872).

Recupera los mensajes más cercanos a la query y los cruza con el sidecar
para devolver texto + sesión + fecha. Complemento semántico del FTS5
liftrario: esta herramienta encuentra por SIGNIFICADO, no por keyword.

Uso:
  hermes_recall.py "cómo quedó el túnel de cloudflare"
  hermes_recall.py "presupuesto del receptor GPS" --k 8 --json
  hermes_recall.py "..." --fts        # agrega hits FTS5 literales (híbrido)
"""
import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_lib import (DIMS, EMBED_URL_DEFAULT, QUERY_PROMPT, fts_query,
                        embed_texts, join_hits, search_hub)

HERMES_HUB = "/mnt/d/splatdb/hermes-hub"
HERMES_STATE_DB = os.path.expanduser(
    "~/.hermes/profiles/deepharness/state.db")


def recall(query, k=5, hub=HERMES_HUB, port=8872,
           embed_url=EMBED_URL_DEFAULT, min_score=None):
    """Query semántica. Devuelve lista de hits enriquecidos + scores."""
    hub_url = f"http://127.0.0.1:{port}"
    qv = embed_texts([query], embed_url=embed_url, dims=DIMS,
                     prompt_name=QUERY_PROMPT)[0]
    hits = search_hub(qv, query, top_k=k, hub_url=hub_url)
    sidecar = sqlite3.connect(f"file:{os.path.join(hub, 'sidecar.db')}?mode=ro",
                              uri=True)
    joined = join_hits(hits, sidecar)
    sidecar.close()
    if min_score is not None:
        joined = [h for h in joined if h["score"] >= min_score]
    return joined


# target → (hub_dir, puerto). 'all' consulta ambos y intercala por score.
TARGETS = {
    "sessions": (HERMES_HUB, 8872),
    "code": ("/mnt/d/splatdb/hermes-hub-code", 8873),
}


def recall_any(query, k=5, target="sessions", **kw):
    if target in TARGETS:
        hub, port = TARGETS[target]
        return recall(query, k=k, hub=hub, port=port, **kw)
    if target == "all":
        merged = []
        for t in TARGETS.values():
            merged.extend(recall(query, k=k, hub=t[0], port=t[1], **kw))
        merged.sort(key=lambda h: -h["score"])
        return merged[:k]
    raise ValueError(f"target inválido: {target}")


def recall_fts(query, k=5, db_path=HERMES_STATE_DB):
    """Búsqueda literal FTS5 sobre la DB viva (complemento híbrido)."""
    tq = fts_query(query)
    if not tq:
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT m.id, m.session_id, m.role, substr(m.content, 1, 400), "
        "bm25(messages_fts) FROM messages_fts f "
        "JOIN messages m ON m.rowid = f.rowid "
        "WHERE messages_fts MATCH ? ORDER BY bm25(messages_fts) LIMIT ?",
        (tq, k)).fetchall()
    con.close()
    return [{"msg_id": r[0], "session_id": r[1], "role": r[2],
             "text": r[3], "bm25": r[4]} for r in rows]


def _fmt(h):
    preview = h["text"][:160].replace("\n", " ")
    if h["kind"] == "repo":
        return (f"[{h['score']:.3f}] [código] {h['repo']}/{h['path']}\n"
                f"    {preview}")
    sid = h["meta"].get("session_id", h["path"])
    date = h["meta"].get("date") or (sid[:8] if sid[:8].isdigit() else "?")
    role = h["meta"].get("role", "?")
    title = h["meta"].get("title", "")
    return (f"[{h['score']:.3f}] {date} {sid[:24]} ({role}) {title}\n"
            f"    {preview}")


def main():
    ap = argparse.ArgumentParser(description="recall semántico de sesiones y código Hermes")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--target", choices=["sessions", "code", "all"], default="sessions",
                    help="sessions=:8872, code=:8873, all=ambos intercalado por score")
    ap.add_argument("--embed-url", default=EMBED_URL_DEFAULT)
    ap.add_argument("--min-score", type=float, default=None,
                    help="corte duro de score (opcional)")
    ap.add_argument("--fts", action="store_true",
                    help="agrega hits FTS5 literales de la DB viva")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    t0 = time.monotonic()
    hits = recall_any(args.query, k=args.k, target=args.target,
                      embed_url=args.embed_url, min_score=args.min_score)
    dt = (time.monotonic() - t0) * 1000
    if args.as_json:
        print(json.dumps({"query": args.query, "target": args.target,
                          "latency_ms": round(dt),
                          "hits": hits}, ensure_ascii=False, indent=2))
        return
    print(f"query: {args.query!r}  [{args.target}] ({dt:.0f} ms, {len(hits)} hits)\n")
    for h in hits:
        print(_fmt(h) + "\n")
    if args.fts:
        print("── FTS5 literal ──")
        for r in recall_fts(args.query, k=args.k):
            print(f"[bm25 {r['bm25']:.2f}] {r['msg_id'][:24]} "
                  f"({r['role']}) {r['text'][:120]}\n")


if __name__ == "__main__":
    main()
