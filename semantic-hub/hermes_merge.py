#!/usr/bin/env python3
"""Merge del part hermes → hermes-hub/sidecar.db + hermes-hub/data/vectors/default.bin.

Aislado a propósito del merge.py del semantic-hub (8871, memorias m2m legacy +
chunks de repos): el hub de sesiones vive en su propio data-dir y puerto (8872),
así un re-merge + restart del hub de sesiones nunca toca el otro.

Formato del bin: u64 nrows | u64 ncols | f32 row-major (igual que merge.py).
"""
import argparse
import os
import sqlite3
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_lib import DIMS, pack_shard_bin

HERMES_HUB = "/mnt/d/splatdb/hermes-hub"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_merge(hub=HERMES_HUB, parts=("hermes",)):
    parts_dir = os.path.join(hub, "parts")
    master_path = os.path.join(hub, "sidecar.db")
    bin_dir = os.path.join(hub, "data", "vectors")
    os.makedirs(bin_dir, exist_ok=True)

    all_blobs = []
    if os.path.exists(master_path):
        os.remove(master_path)
    con = sqlite3.connect(master_path)
    con.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, repo TEXT, "
                "path TEXT, kind TEXT, meta TEXT, text TEXT, vec BLOB)")
    con.execute("CREATE TABLE parts (part TEXT, first_rowid INTEGER, n INTEGER)")
    rowid = 0
    for part in parts:
        p = os.path.join(parts_dir, f"{part}.db")
        src = sqlite3.connect(p)
        rows = src.execute("SELECT rowid, repo, path, kind, meta, text, vec "
                           "FROM chunks ORDER BY rowid").fetchall()
        first = rowid + 1
        for _, repo, path, kind, meta, text, vec in rows:
            rowid += 1
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (rowid, repo, path, kind, meta, text, vec))
            all_blobs.append(bytes(vec))
        con.execute("INSERT INTO parts VALUES (?,?,?)", (part, first, len(rows)))
        src.close()
        log(f"  {part}: {len(rows)} chunks (rowids {first}..{rowid})")
    con.commit()
    con.close()

    bin_path = os.path.join(bin_dir, "default.bin")
    blob = pack_shard_bin(all_blobs)
    with open(bin_path, "wb") as f:
        f.write(blob)
    log(f"MASTER: {len(all_blobs)} chunks → {master_path} + "
        f"{bin_path} ({os.path.getsize(bin_path)/1e6:.1f} MB)")
    return len(all_blobs)


def main():
    ap = argparse.ArgumentParser(description="Merge del part hermes → sidecar + shard bin")
    ap.add_argument("--hub", default=HERMES_HUB)
    args = ap.parse_args()
    run_merge(hub=args.hub)


if __name__ == "__main__":
    main()
