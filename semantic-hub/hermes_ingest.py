#!/usr/bin/env python3
"""Ingesta incremental de sesiones Hermes → part sqlite del hermes-hub.

Lee state.db de Hermes (SOLO LECTURA — jamás escribe en la DB viva),
filtra mensajes de conversación real, embebe con embeddinggemma-2 (:8860)
y agrega al part hermes.db con metadatos para el join post-búsqueda.

Incremental: guarda MAX(messages.rowid) procesado en la tabla `state` del
part. Corridas siguientes solo procesan mensajes nuevos. Dedup por hash de
texto (re-envíos idénticos no duplican vectores).

Uso:
  hermes_ingest.py                     # corrida incremental normal
  hermes_ingest.py --rebuild           # wipe + reingesta completa
  hermes_ingest.py --db /tmp/fake.db --limit 50   # test con DB falsa
"""
import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_lib import (DIMS, EMBED_URL_DEFAULT, doc_hash, embed_texts,
                        meta_json, parse_message_row, vec_blob)

HERMES_STATE_DB = os.path.expanduser(
    "~/.hermes/profiles/deepharness/state.db")
HERMES_HUB = "/mnt/d/splatdb/hermes-hub"
BATCH = 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def open_source_ro(db_path):
    """Abre la DB fuente en modo read-only (la DB de Hermes está viva)."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def ensure_part_db(part_path, rebuild=False):
    """Crea/abre el part. Devuelve conexión y el last_rowid procesado."""
    os.makedirs(os.path.dirname(part_path), exist_ok=True)
    fresh = rebuild or not os.path.exists(part_path)
    if rebuild and os.path.exists(part_path):
        os.remove(part_path)
    con = sqlite3.connect(part_path)
    con.execute("CREATE TABLE IF NOT EXISTS chunks (rowid INTEGER PRIMARY KEY, "
                "repo TEXT, path TEXT, kind TEXT, meta TEXT, text TEXT, vec BLOB)")
    con.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS doc_hashes (hash TEXT PRIMARY KEY)")
    con.commit()
    if fresh:
        con.execute("INSERT OR REPLACE INTO state VALUES ('last_rowid', '0')")
        con.commit()
        return con, 0
    row = con.execute("SELECT value FROM state WHERE key='last_rowid'").fetchone()
    return con, int(row[0]) if row else 0


def fetch_new_messages(con_src, after_rowid, limit=None):
    """Mensajes nuevos ya filtrados por parse_message_row."""
    q = ("SELECT m.rowid, m.id, m.session_id, m.role, m.content, s.title "
         "FROM messages m JOIN sessions s ON s.id = m.session_id "
         "WHERE m.rowid > ? AND m.role IN ('user','assistant') AND m.active = 1 "
         "ORDER BY m.rowid")
    if limit:
        q += f" LIMIT {int(limit)}"
    for row in con_src.execute(q, (after_rowid,)):
        doc = parse_message_row(row)
        if doc is not None:
            yield doc


def run_ingest(db_path=HERMES_STATE_DB, hub=HERMES_HUB, rebuild=False,
               limit=None, embed_url=EMBED_URL_DEFAULT, batch=BATCH, log_fn=log):
    """Corrida de ingesta. Devuelve (insertados, saltados_dedup, total_part)."""
    part_path = os.path.join(hub, "parts", "hermes.db")
    con, last_rowid = ensure_part_db(part_path, rebuild=rebuild)
    con_src = open_source_ro(db_path)
    max_seen = last_rowid

    pending, inserted, dup = [], 0, 0

    def flush(pending):
        nonlocal inserted, dup
        if not pending:
            return
        vecs = embed_texts([d["text"] for d in pending], embed_url=embed_url,
                           dims=DIMS, prompt_name=None)
        rows = []
        for doc, v in zip(pending, vecs):
            h = doc_hash(doc["text"])
            try:
                con.execute("INSERT INTO doc_hashes VALUES (?)", (h,))
            except sqlite3.IntegrityError:
                dup += 1
                continue
            rows.append((doc["session_id"], doc["msg_id"], "hermes/session",
                         meta_json(doc), doc["text"], sqlite3.Binary(vec_blob(v))))
        if rows:
            con.executemany(
                "INSERT INTO chunks (repo, path, kind, meta, text, vec) "
                "VALUES (?,?,?,?,?,?)", rows)
            inserted += len(rows)
        con.commit()

    for doc in fetch_new_messages(con_src, last_rowid, limit=limit):
        max_seen = max(max_seen, doc["rowid"])
        pending.append(doc)
        if len(pending) >= batch:
            flush(pending)
            log_fn(f"  {inserted} nuevos, {dup} dup (rowid>={pending[-1]['rowid']})")
            pending = []
    flush(pending)

    con.execute("INSERT OR REPLACE INTO state VALUES ('last_rowid', ?)",
                (str(max_seen),))
    total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.commit()
    con.close()
    con_src.close()
    return inserted, dup, total


def main():
    ap = argparse.ArgumentParser(
        description="Ingesta incremental de sesiones Hermes → part sqlite")
    ap.add_argument("--db", default=HERMES_STATE_DB, help="state.db fuente (RO)")
    ap.add_argument("--hub", default=HERMES_HUB, help="dir del hermes-hub")
    ap.add_argument("--rebuild", action="store_true", help="wipe del part")
    ap.add_argument("--limit", type=int, default=None, help="tope de mensajes a leer")
    ap.add_argument("--embed-url", default=EMBED_URL_DEFAULT)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    t0 = time.monotonic()
    inserted, dup, total = run_ingest(
        db_path=args.db, hub=args.hub, rebuild=args.rebuild,
        limit=args.limit, embed_url=args.embed_url, batch=args.batch)
    dt = time.monotonic() - t0
    log(f"LISTO: {inserted} nuevos, {dup} duplicados, total part={total} ({dt:.0f}s)")
    if inserted == 0 and dup == 0:
        log("sin mensajes nuevos — nada que hacer")


if __name__ == "__main__":
    main()
