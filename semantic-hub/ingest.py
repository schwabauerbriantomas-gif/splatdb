#!/usr/bin/env python3
"""Ingesta semántica: repos/documentos → embeddings embeddinggemma-2 → sidecar sqlite + shard bin.

Cada corrida escribe en SU PROPIO sidecar (parts/<name>.db) — sin pisadas entre hijos.
El padre hace merge de los parts → hub/sidecar.db + hub/data/vectors/default.bin.

Uso:
  ingest.py --name parteA --kind repo --repo /mnt/d/vuupta [--repo /mnt/d/splatdb ...]
  ingest.py --name memoria --kind memory --m2m /mnt/d/m2m-data/data/metadata/metadata.db

Formato del shard bin (src/storage/persistence.rs de splatdb):
  u64 le nrows | u64 le ncols | nrows*ncols f32 le row-major
"""
import argparse
import hashlib
import json
import os
import sqlite3
import struct
import sys
import time
import urllib.request

EMB_URL = "http://127.0.0.1:8860/v1/embeddings"
HUB = "/mnt/d/splatdb/semantic-hub"
DIMS = 768

SKIP_DIRS = {".git", ".github", "node_modules", "target", "dist", "build", "out",
             "__pycache__", ".venv", "venv", ".next", "vendor", ".history",
             "backups", ".idea", ".vscode", ".gradle", "gradle", "site-packages",
             ".pytest_cache", ".mypy_cache", ".ruff_cache", "emb-test", ".dart_tool",
             "Pods", ".sx", ".fvm", "coverage", ".turbo", ".cache", ".parcel-cache"}
# Archivos con secretos: nunca indexar (quedan en el sidecar y son buscables)
SECRET_PATTERNS = ("service_account", "credentials", "secret", "id_rsa", "id_ed25519",
                   ".pem", ".p12", ".pfx", ".key", ".env", "token", "keystore")
SECRET_CONTENT = ("BEGIN RSA PRIVATE KEY", "BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY",
                  "BEGIN EC PRIVATE KEY", "AIzaSy", "ghp_", "sk-ant-", "xoxb-")
TEXT_EXT = {".py", ".rs", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs", ".md", ".txt",
            ".rst", ".toml", ".yaml", ".yml", ".json", ".sh", ".bash", ".cmd", ".ps1",
            ".sql", ".html", ".htm", ".css", ".scss", ".c", ".h", ".cpp", ".hpp", ".cc",
            ".hh", ".cs", ".kt", ".kts", ".java", ".go", ".dart", ".swift", ".rb", ".php",
            ".lua", ".xml", ".gradle", ".properties", ".ini", ".cfg", ".conf", ".proto",
            ".cmake", ".mk", ".groovy", ".kdl", ".ron"}
EXACT_NAMES = {"dockerfile", "makefile", "cmakelists.txt", "license", "cargo.lock.skip"}
MAX_FILE_CHARS = 40000
CHUNK_CHARS = 1400
CHUNK_OVERLAP_CHARS = 150
DEFAULT_CAP = 800


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def embed_batch(texts, retries=3):
    body = json.dumps({"input": texts, "dimensions": DIMS}).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(EMB_URL, data=body,
                                         headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=180).read())
            return [x["embedding"] for x in d["data"]]
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise  # error de cliente: reintentar no ayuda
            last = e
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def embed_safe(items):
    """items = [(meta, text), ...]. Biseca ante fallas y salta el culpable.
    → (vecs, items_kept, skipped) — alineados por posición."""
    texts = [t for _, t in items]
    try:
        return embed_batch(texts), items, 0
    except Exception as e:
        if len(items) == 1:
            log(f"  chunk salteado (embed fallo): {str(e)[:70]} | muestra: {items[0][1][:50]!r}")
            return [], [], 1
        mid = len(items) // 2
        v1, k1, s1 = embed_safe(items[:mid])
        v2, k2, s2 = embed_safe(items[mid:])
        return v1 + v2, k1 + k2, s1 + s2


def walk_files(root):
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".git")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if fn.lower() in EXACT_NAMES:
                pass
            elif ext == ".json":
                pass  # permitido con cap de tamaño abajo
            elif ext not in TEXT_EXT:
                continue
            yield os.path.join(dirpath, fn)


def chunk_text(text):
    if len(text) <= CHUNK_CHARS:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + CHUNK_CHARS - 200, end)
            if nl > start:
                end = nl + 1
        piece = text[start:end]
        if piece.strip():
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - CHUNK_OVERLAP_CHARS if end > CHUNK_OVERLAP_CHARS else end
    return chunks


def iter_repo_chunks(repo_root, cap):
    seen_hashes = set()
    n_files = n_chunks = 0
    for path in walk_files(repo_root):
        base = os.path.basename(path).lower()
        if any(p in base for p in SECRET_PATTERNS):
            continue
        try:
            if os.path.getsize(path) > 400_000:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_FILE_CHARS)
        except OSError:
            continue
        if not text.strip():
            continue
        n_files += 1
        rel = os.path.relpath(path, repo_root)
        for ch in chunk_text(text):
            if any(s in ch for s in SECRET_CONTENT):
                continue
            h = hashlib.blake2b(ch.encode(), digest_size=12).digest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            yield rel, ch
            n_chunks += 1
            if n_chunks >= cap:
                log(f"  cap {cap} chunks alcanzado en {repo_root}")
                return
    log(f"  {repo_root}: {n_files} archivos, {n_chunks} chunks únicos")


def iter_m2m_chunks(db_path):
    con = sqlite3.connect(db_path)
    cur = con.execute("SELECT vector_idx, metadata, document, deleted, created_at "
                      "FROM documents WHERE deleted = 0 ORDER BY vector_idx")
    for vec_idx, meta, doc, deleted, created in cur:
        text = (doc or "").strip() or (meta or "").strip()
        if not text:
            text = str(vec_idx)
        yield (f"m2m/{str(created)[:10] if created else 'sindate'}", text[:6000],
               json.dumps({"m2m_id": vec_idx, "created": created}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nombre del part (su sidecar)")
    ap.add_argument("--kind", default="repo", choices=["repo", "memory"])
    ap.add_argument("--repo", action="append", default=[], help="raíz de repo (repetible)")
    ap.add_argument("--m2m", help="metadata.db de m2m para migrar")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help="chunks por repo")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    parts_dir = os.path.join(HUB, "parts")
    os.makedirs(parts_dir, exist_ok=True)
    db_path = os.path.join(parts_dir, f"{args.name}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, repo TEXT, path TEXT, "
                "kind TEXT, meta TEXT, text TEXT, vec BLOB)")
    con.commit()

    batch_texts, batch_meta = [], []

    def flush():
        if not batch_texts:
            return
        vecs, kept, skipped = embed_safe(list(zip(batch_meta, batch_texts)))
        if skipped:
            log(f"  {skipped} chunk(s) salteado(s) por fallo de embed")
        rows = [(meta[0], meta[1], meta[2], meta[3], t, sqlite3.Binary(struct.pack(f"<{DIMS}f", *v)))
                for (meta, t), v in zip(kept, vecs)]
        con.executemany("INSERT INTO chunks (repo, path, kind, meta, text, vec) VALUES (?,?,?,?,?,?)", rows)
        con.commit()
        batch_texts.clear()
        batch_meta.clear()

    total = 0
    t0 = time.monotonic()
    if args.m2m:
        log(f"migrando m2m desde {args.m2m}")
        for repo, text, meta in iter_m2m_chunks(args.m2m):
            batch_texts.append(text)
            batch_meta.append((repo, "memory/origen", args.kind, meta))
            total += 1
            if len(batch_texts) >= args.batch:
                flush()
                if total % 320 == 0:
                    log(f"  {total} docs...")
        flush()
    for spec in args.repo:
        if ":" in spec and os.path.exists(spec.rsplit(":", 1)[0]):
            repo_path, override = spec.rsplit(":", 1)
        else:
            repo_path, override = spec, None
        repo = os.path.abspath(repo_path)
        name = override or os.path.basename(repo.rstrip("/"))
        log(f"indexando {name} ({repo})")
        for rel, ch in iter_repo_chunks(repo, args.cap):
            batch_texts.append(ch)
            batch_meta.append((name, rel, args.kind, ""))
            total += 1
            if len(batch_texts) >= args.batch:
                flush()
                if total % 400 == 0:
                    log(f"  {total} chunks...")
        flush()

    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    dt = time.monotonic() - t0
    log(f"LISTO part={args.name}: {n} chunks en {dt:.0f}s → {db_path}")


if __name__ == "__main__":
    main()
