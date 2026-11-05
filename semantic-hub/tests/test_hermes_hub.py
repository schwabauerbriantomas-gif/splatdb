#!/usr/bin/env python3
"""Tests unitarios del hub de memoria Hermes — solo stdlib, sin GPU ni servicios.

El embed server se mockea con un http.server en-thread que devuelve vectores
deterministas: cubre ingesta end-to-end (fake state.db → part), merge
(formato del bin + sidecar) y dedup/incrementalidad, todo en CI runner.
"""
import http.server
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_lib import (DIMS, parse_message_row, pack_shard_bin,
                        read_shard_bin_header, vec_blob, fts_query,
                        join_hits, meta_json)


def make_vec(seed):
    """Vector determinista normalizado-ish (suficiente para el test)."""
    v = [((seed + i) % 13) / 13.0 for i in range(DIMS)]
    return v


class FakeEmbedHandler(http.server.BaseHTTPRequestHandler):
    """POST /v1/embeddings → un vector por texto del batch."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        texts = body["input"]
        out = [{"embedding": make_vec(len(t) % 97 + i)} for i, t in enumerate(texts)]
        payload = json.dumps({"data": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def make_source_db(path, messages):
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT);"
        "CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT,"
        " content TEXT, active INTEGER);"
        "INSERT INTO sessions VALUES ('s1', 'sesion de prueba');")
    for i, (role, content) in enumerate(messages):
        con.execute("INSERT INTO messages VALUES (?,?,?,?,1)",
                    (f"m{i}", "s1", role, content))
    con.commit()
    con.close()


LONG = ("texto de conversacion real con suficiente contenido semantico "
        "para pasar el filtro de largo minimo del indice ") * 3


class TestParseFilters(unittest.TestCase):
    def test_valid_message_passes(self):
        doc = parse_message_row((1, "m1", "s1", "user", LONG, "titulo"))
        self.assertIsNotNone(doc)
        self.assertEqual(doc["text"], LONG.strip())

    def test_short_message_filtered(self):
        self.assertIsNone(parse_message_row((1, "m1", "s1", "user", "corto", "")))

    def test_notification_prefixes_filtered(self):
        for prefix in ("[IMPORTANT: algo", "[ASYNC DELEGATION task",
                       "[Background process proc_1"):
            self.assertIsNone(
                parse_message_row((1, "m1", "s1", "assistant", prefix + LONG, "")))

    def test_json_blob_filtered(self):
        self.assertIsNone(
            parse_message_row((1, "m1", "s1", "assistant", '{"success": true, "data": 2}', "")))

    def test_tool_role_filtered(self):
        self.assertIsNone(parse_message_row((1, "m1", "s1", "tool", LONG, "")))

    def test_long_text_truncated(self):
        doc = parse_message_row((1, "m1", "s1", "user", LONG * 30, ""))
        self.assertIsNotNone(doc)
        self.assertLessEqual(len(doc["text"]), 1500)


class TestShardBin(unittest.TestCase):
    def test_roundtrip_header(self):
        blobs = [sqlite3.Binary(bytes(vec_blob(make_vec(i)))) for i in range(3)]
        data = pack_shard_bin(blobs)
        self.assertEqual(len(data), 16 + 3 * DIMS * 4)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            p = f.name
        try:
            self.assertEqual(read_shard_bin_header(p), (3, DIMS))
        finally:
            os.remove(p)

    def test_vec_blob_wrong_dim_raises(self):
        with self.assertRaises(ValueError):
            vec_blob([0.0] * 10)


class TestFtsQuery(unittest.TestCase):
    def test_or_of_quoted_terms(self):
        q = fts_query("cómo quedó el túnel de Cloudflare")
        self.assertIn('"túnel"', q)
        self.assertIn('"cloudflare"', q)
        self.assertNotIn('"cómo"', q)   # stopword
        self.assertNotIn('"el"', q)     # stopword
        self.assertIn(" OR ", q)

    def test_max_terms(self):
        n = len(fts_query("a b c d e f g h i j k").split(" OR "))
        self.assertLessEqual(n, 8)

    def test_empty_after_stopwords(self):
        self.assertEqual(fts_query("de la que el"), "")


class TestJoinHits(unittest.TestCase):
    def test_join_and_skip_missing(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, repo TEXT,"
                    " path TEXT, kind TEXT, meta TEXT, text TEXT, vec BLOB)")
        con.execute("INSERT INTO chunks VALUES (1,'r','p','k','{\"date\": \"20261105\"}','hola',NULL)")
        hits = [{"index": 0, "score": 0.9}, {"index": 5, "score": 0.8}]
        joined = join_hits(hits, con)
        self.assertEqual(len(joined), 1)  # index 5 sin fila → descartado
        self.assertEqual(joined[0]["text"], "hola")
        self.assertEqual(joined[0]["meta"]["date"], "20261105")


class TestIngestE2E(unittest.TestCase):
    """Ingesta completa contra embed server mockeado: fake DB → part."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hermes-test-")
        self.src = os.path.join(self.dir, "src.db")
        self.hub = os.path.join(self.dir, "hub")
        srv = http.server.HTTPServer(("127.0.0.1", 0), FakeEmbedHandler)
        self.port = srv.server_address[1]
        self.thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self.thread.start()
        self.srv = srv

    def tearDown(self):
        self.srv.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run_ingest(self, rebuild=False):
        import hermes_ingest
        inserted, dup, total = hermes_ingest.run_ingest(
            db_path=self.src, hub=self.hub, rebuild=rebuild,
            embed_url=f"http://127.0.0.1:{self.port}/v1/embeddings",
            batch=8, log_fn=lambda *a: None)
        return inserted, dup, total

    def test_incremental_and_dedup(self):
        make_source_db(self.src, [("user", LONG), ("assistant", LONG + "respuesta distinta")])
        ins, dup, total = self._run_ingest(rebuild=True)
        self.assertEqual((ins, total), (2, 2))

        # nuevo mensaje → solo ese entra
        con = sqlite3.connect(self.src)
        con.execute("INSERT INTO messages VALUES ('m9','s1','user',?,1)",
                    ("mensaje nuevo posterior a la primera ingesta " + LONG[:100],))
        con.commit()
        con.close()
        ins, dup, total = self._run_ingest()
        self.assertEqual(ins, 1)
        self.assertEqual(total, 3)

        # re-corrida sin mensajes nuevos → cero
        ins, dup, total = self._run_ingest()
        self.assertEqual(ins, 0)

        # texto duplicado (re-envío) → dedup por hash
        con = sqlite3.connect(self.src)
        con.execute("INSERT INTO messages VALUES ('m10','s1','user',?,1)", (LONG,))
        con.commit()
        con.close()
        ins, dup, total = self._run_ingest()
        self.assertEqual((ins, dup), (0, 1))

    def test_merge_writes_consistent_master(self):
        import hermes_merge
        make_source_db(self.src, [("user", LONG), ("assistant", LONG + " dos")])
        self._run_ingest(rebuild=True)
        n = hermes_merge.run_merge(hub=self.hub)
        self.assertEqual(n, 2)
        nrows, ncols = read_shard_bin_header(
            os.path.join(self.hub, "data", "vectors", "default.bin"))
        self.assertEqual((nrows, ncols), (2, DIMS))
        con = sqlite3.connect(os.path.join(self.hub, "sidecar.db"))
        self.assertEqual(con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 2)
        meta = json.loads(con.execute(
            "SELECT meta FROM chunks WHERE rowid=1").fetchone()[0])
        self.assertEqual(meta["session_id"], "s1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
