#!/usr/bin/env python3
"""Merge de parts → hub/sidecar.db + hub/data/vectors/default.bin (formato splatdb).

Uso: merge.py part_a part_b ...   (orden = orden de rowids en el master)
"""
import glob
import os
import sqlite3
import struct
import sys
import time

HUB = "/mnt/d/splatdb/semantic-hub"
DIMS = 768


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    parts = sys.argv[1:] or sorted(os.path.basename(p)[:-3] for p in
                                   glob.glob(os.path.join(HUB, "parts", "*.db")))
    master = os.path.join(HUB, "sidecar.db")
    if os.path.exists(master):
        os.remove(master)
    con = sqlite3.connect(master)
    con.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, repo TEXT, path TEXT, "
                "kind TEXT, meta TEXT, text TEXT, vec BLOB)")
    con.execute("CREATE TABLE parts (part TEXT, first_rowid INTEGER, n INTEGER)")

    all_vecs = []
    rowid = 0
    for part in parts:
        p = os.path.join(HUB, "parts", f"{part}.db")
        src = sqlite3.connect(p)
        rows = src.execute("SELECT rowid, repo, path, kind, meta, text, vec FROM chunks "
                           "ORDER BY rowid").fetchall()
        first = rowid + 1
        for _, repo, path, kind, meta, text, vec in rows:
            rowid += 1
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (rowid, repo, path, kind, meta, text, vec))
            all_vecs.append(vec)
        con.execute("INSERT INTO parts VALUES (?,?,?)", (part, first, len(rows)))
        src.close()
        log(f"  {part}: {len(rows)} chunks (rowids {first}..{rowid})")
    con.commit()

    n = len(all_vecs)
    bin_dir = os.path.join(HUB, "data", "vectors")
    os.makedirs(bin_dir, exist_ok=True)
    bin_path = os.path.join(bin_dir, "default.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<Q", DIMS))
        for v in all_vecs:
            f.write(v)
    size_mb = os.path.getsize(bin_path) / 1e6
    log(f"MASTER: {n} chunks → {master} + {bin_path} ({size_mb:.1f} MB)")
    con.close()


if __name__ == "__main__":
    main()
