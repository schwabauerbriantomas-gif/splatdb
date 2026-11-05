#!/usr/bin/env python3
"""hermes_lib — lógica pura del hub de memoria de sesiones Hermes.

Sin efectos secundarios y solo stdlib: todo lo testeable vive acá para que
el CI pueda validarlo sin GPU, sin servicios y sin datos reales.

Formato del shard bin (idéntico a merge.py / src/storage/persistence.rs):
  u64 le nrows | u64 le ncols | nrows*ncols f32 le row-major
"""
import hashlib
import json
import re
import struct
import urllib.request

DIMS = 768
MAX_TEXT_CHARS = 1500          # techo por mensaje (mide igual que el bench)
MIN_TEXT_CHARS = 60            # piso: debajo es ruido de turno
EMBED_URL_DEFAULT = "http://127.0.0.1:8860/v1/embeddings"
QUERY_PROMPT = "Retrieval-query"   # prompt oficial embeddinggemma para queries

# Notificaciones de sistema que llegan como role=assistant y ensucian el corpus.
NOTIFICATION_PREFIXES = (
    "[IMPORTANT:",
    "[ASYNC DELEGATION",
    "[OUT-OF-BAND USER MESSAGE",
    "[Command timed out",
    "[Background process",
)

STOPWORDS = set(
    "de la el los las un una unas unos y o u e a en que por con para sin al lo "
    "su sus es ser fue era como mas muy ya sino entre sobre tras durante segun "
    "ante bajo desde hasta cual cuales quien cuyo donde ademas pero porque "
    "este esta estos estas ese esa esos esas "
    "cómo dónde qué quién cuándo cuánto cuántos más también después antes".split()
)


def parse_message_row(row):
    """row = (rowid, msg_id, session_id, role, content, title) → dict | None.

    Filtros: solo user/assistant activos, largo razonable, sin notificaciones
    de sistema ni blobs JSON puros (contenido thin que degrada el índice).
    """
    rowid, msg_id, session_id, role, content, title = row
    if role not in ("user", "assistant"):
        return None
    if not content:
        return None
    text = content.strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    if text.startswith(NOTIFICATION_PREFIXES):
        return None
    if text[0] in "{[" :
        return None  # blob JSON / lista: sin prosa semántica útil
    return {
        "rowid": rowid,
        "msg_id": msg_id,
        "session_id": session_id,
        "role": role,
        "title": title or "",
        "text": text[:MAX_TEXT_CHARS],
    }


def doc_hash(text):
    """Hash de dedup: el mismo texto re-enviado no entra dos veces al índice."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def meta_json(doc):
    """Metadata JSON que viaja en el sidecar para el join post-búsqueda."""
    mid = str(doc["msg_id"]) if doc["msg_id"] is not None else ""
    head = mid[:8]
    date = head if re.fullmatch(r"\d{8}", head) else ""
    return json.dumps({
        "message_id": mid,
        "session_id": str(doc["session_id"]),
        "role": doc["role"],
        "title": doc["title"],
        "date": date,
    }, ensure_ascii=False)


def pack_shard_bin(vec_blobs):
    """Lista de blobs (n*768 f32 LE) → bytes del formato splatdb."""
    n = len(vec_blobs)
    out = [struct.pack("<Q", n), struct.pack("<Q", DIMS)]
    out.extend(vec_blobs)
    return b"".join(out)


def read_shard_bin_header(path):
    """(nrows, ncols) del header del shard bin."""
    with open(path, "rb") as f:
        nrows, ncols = struct.unpack("<QQ", f.read(16))
    return nrows, ncols


def vec_blob(vec):
    """Lista/tupla de floats → blob f32 LE de DIMS elementos."""
    if len(vec) != DIMS:
        raise ValueError(f"embedding de {len(vec)} dims, se esperaban {DIMS}")
    return sqlite_binary(struct.pack(f"<{DIMS}f", *vec))


def sqlite_binary(data):
    """Wrapper explícito para documentar que el blob va crudo a sqlite3."""
    return memoryview(data)


def fts_query(q, max_terms=8):
    """Query FTS5 estilo keyword-search: términos únicos con OR, sin stopwords."""
    terms = []
    for t in re.findall(r"[a-z0-9áéíóúñü]{3,}", q.lower()):
        if t in STOPWORDS or t in terms:
            continue
        terms.append(t)
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{t}"' for t in terms)


def embed_texts(texts, embed_url=EMBED_URL_DEFAULT, dims=DIMS,
                prompt_name=None, retries=3, timeout=300):
    """Embeddings vía server local. Devuelve lista de listas de floats.

    Reintenta con backoff en errores de server (5xx/red); los 4xx no se
    reintentan (el payload no va a mejorar solo).
    """
    import time
    import urllib.error
    from urllib.request import Request, urlopen
    body = {"input": texts, "model": "embeddinggemma-2"}
    if prompt_name:
        body["prompt_name"] = prompt_name
    if dims:
        body["dimensions"] = dims
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = Request(data=data, url=embed_url,
                          headers={"Content-Type": "application/json"})
            resp = json.loads(urlopen(req, timeout=timeout).read())
            return [d["embedding"] for d in resp["data"]]
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                raise
            last = e
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # red, timeout, JSON roto
            last = e
            time.sleep(2 * (attempt + 1))
    raise last if last is not None else RuntimeError("embed_texts: fallo sin excepción")


def search_hub(query_embedding, query_text, top_k, hub_url,
               timeout=60, retries=2):
    """POST /search al splatdb. Devuelve [{index, score}, ...]."""
    import time
    from urllib.request import Request, urlopen
    body = json.dumps({
        "query": query_text,
        "embedding": query_embedding,
        "top_k": top_k,
    }).encode()
    last = None
    for attempt in range(retries):
        try:
            req = Request(data=body, url=f"{hub_url}/search",
                          headers={"Content-Type": "application/json"})
            resp = json.loads(urlopen(req, timeout=timeout).read())
            return resp.get("results", [])
        except Exception as e:
            last = e
            time.sleep(1 + attempt)
    raise last if last is not None else RuntimeError("search_hub: fallo sin excepción")


def join_hits(hits, sidecar_conn):
    """Cruza [{index, score}] con el sidecar (chunks.rowid = index + 1).

    Devuelve [{index, score, text, repo, path, kind, meta(dict)}] ordenado
    por score descendente. Los índices sin fila en el sidecar se descartan
    (jamás inventar texto).
    """
    out = []
    for h in hits:
        row = sidecar_conn.execute(
            "SELECT repo, path, kind, meta, text FROM chunks WHERE rowid = ?",
            (h["index"] + 1,)).fetchone()
        if row is None:
            continue
        repo, path, kind, meta, text = row
        try:
            meta_d = json.loads(meta) if meta else {}
        except (ValueError, TypeError):
            meta_d = {}
        out.append({
            "index": h["index"],
            "score": h.get("score", 0.0),
            "repo": repo, "path": path, "kind": kind,
            "meta": meta_d, "text": text,
        })
    return out
