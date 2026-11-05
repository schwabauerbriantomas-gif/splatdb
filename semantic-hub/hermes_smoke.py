#!/usr/bin/env python3
"""Smoke test end-to-end del hub de memoria Hermes.

Valida, contra servicios reales:
  1. embed server (:8860) y hub (:8872) sanos
  2. sidecar y shard bin consistentes (mismo número de docs)
  3. recall semántico: queries de paráfrasis (sin compartir keywords con
     el target) encuentran la sesión esperada en top-k

Las tripletas (query, sesión esperada, k) están tomadas del benchmark
medido el 2026-11-05 (corpus real, MRR sesión 0.60 vs 0.27 FTS). Si el
corpus crece con sesiones más nuevas, el rank puede moverse: el smoke
falla solo si la sesión correcta sale del top-k.

Exit 0 = PASS. Exit 1 = FAIL (detalla cuál check falló).
"""
import json
import os
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hermes_lib import EMBED_URL_DEFAULT, read_shard_bin_header

HERMES_HUB = "/mnt/d/splatdb/hermes-hub"
HUB_PORT = 8872

# (nombre, query, substring esperado en session_id, k)
CASES = [
    ("tunel_autoupdate",
     "reiniciar el demonio que publica los sitios y dejarlo actualizandose solo",
     "20261026_191031", 8),
    ("token_crontab",
     "revision de seguridad: clave secreta expuesta en tareas programadas del sistema",
     "20260911_120551", 5),
    ("fuentes_norte",
     "que provincias del norte quedaron verificadas para el monitor de compras publicas",
     "20260923_191657_e81156", 5),
]


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    failures = 0
    print("smoke hermes-hub:")

    # 1. servicios
    for name, url in (("embed-server", f"{EMBED_URL_DEFAULT.split('/v1')[0]}/health"),
                      ("hub", f"http://127.0.0.1:{HUB_PORT}/health")):
        try:
            resp = json.loads(urllib.request.urlopen(url, timeout=5).read())
            ok = resp.get("status") == "ok"
            check(f"health {name}", ok, json.dumps(resp)[:80])
            failures += 0 if ok else 1
        except Exception as e:
            failures += 1
            check(f"health {name}", False, str(e)[:80])

    # 2. consistencia sidecar/bin (ambos hubs)
    for label, hub in (("sesiones", HERMES_HUB),
                       ("código", "/mnt/d/splatdb/hermes-hub-code")):
        ndocs = -1
        try:
            bin_path = os.path.join(hub, "data", "vectors", "default.bin")
            nrows, ncols = read_shard_bin_header(bin_path)
            sidecar = sqlite3.connect(
                f"file:{os.path.join(hub, 'sidecar.db')}?mode=ro", uri=True)
            ndocs = sidecar.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            sidecar.close()
            ok = nrows == ndocs and ncols == 768
            check(f"sidecar/bin consistentes ({label})", ok,
                  f"bin={nrows}x{ncols}, sidecar={ndocs}")
            failures += 0 if ok else 1
        except Exception as e:
            failures += 1
            check(f"sidecar/bin consistentes ({label})", False, str(e)[:80])

    # 4. drift part vs sidecar: si el part creció sin merge, el índice queda
    #    viejo (bug real detectado: ciclo sin merge de sesiones)
    try:
        part_db = os.path.join(hub, "parts")
        parts = [f for f in os.listdir(part_db) if f.endswith(".db")]
        con = sqlite3.connect(f"file:{os.path.join(part_db, parts[0])}?mode=ro",
                              uri=True)
        npart = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        con.close()
        ok = npart == ndocs
        check(f"part/sidecar sin drift ({label})", ok,
              f"part={npart}, sidecar={ndocs}" + ("" if ok else " — falta merge"))
        failures += 0 if ok else 1
    except Exception as e:
        failures += 1
        check(f"part/sidecar sin drift ({label})", False, str(e)[:80])

    # 3. health del hub de código
    try:
        resp = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:8873/health", timeout=5).read())
        ok = resp.get("status") == "ok"
        check("health hub código (:8873)", ok, json.dumps(resp)[:80])
        failures += 0 if ok else 1
    except Exception as e:
        failures += 1
        check("health hub código (:8873)", False, str(e)[:80])

    # 4. recall semántico (paráfrasis → sesión esperada en top-k)
    from hermes_recall import recall
    for name, query, sid_substr, k in CASES:
        try:
            t0 = time.monotonic()
            hits = recall(query, k=k)
            dt = (time.monotonic() - t0) * 1000
            got = [h["meta"].get("session_id", "") for h in hits]
            rank = next((i + 1 for i, s in enumerate(got) if sid_substr in s), None)
            ok = rank is not None
            check(f"recall {name}", ok,
                  f"rank={rank}/{k} ({dt:.0f} ms)" if ok
                  else f"sesión {sid_substr} fuera de top-{k}: {got[:3]}")
            failures += 0 if ok else 1
        except Exception as e:
            failures += 1
            check(f"recall {name}", False, str(e)[:80])

    # 5. recall de código: en el hub puro de código, top-3 debe ser kind=repo
    try:
        from hermes_recall import recall_any
        t0 = time.monotonic()
        hits = recall_any("funcion que empaqueta vectores en archivo binario con "
                          "header de counts y dimensiones", k=3, target="code")
        dt = (time.monotonic() - t0) * 1000
        ok = bool(hits) and all(h["kind"] == "repo" for h in hits[:3])
        check("recall código (top-3 = kind repo)", ok,
              f"kinds={[h['kind'] for h in hits[:3]]} ({dt:.0f} ms)")
        failures += 0 if ok else 1
    except Exception as e:
        failures += 1
        check("recall código (top-3 = kind repo)", False, str(e)[:80])

    # 5. sanidad de la métrica: la distancia del literal DEBE ser menor que
    #    la de basura (detecta inversiones de semántica score/distance)
    try:
        from hermes_lib import embed_texts, search_hub, QUERY_PROMPT, DIMS
        import sqlite3 as _sq
        _con = _sq.connect(f"file:{os.path.join(HERMES_HUB, 'sidecar.db')}?mode=ro",
                           uri=True)
        _lit = _con.execute(
            "SELECT text FROM chunks WHERE meta LIKE '%20261026_191031%' "
            "LIMIT 1").fetchone()[0][:120]
        _con.close()
        d = {}
        for lbl, q in (("literal", _lit),
                       ("basura", "receta de tallarines con estofado")):
            _qv = embed_texts([q], dims=DIMS, prompt_name=QUERY_PROMPT)[0]
            d[lbl] = search_hub(_qv, q, top_k=1,
                                hub_url="http://127.0.0.1:8872")[0]["score"]
        ok = d["literal"] < d["basura"]
        check("métrica: literal < basura (distancia)", ok,
              f"literal={d['literal']:.3f}, basura={d['basura']:.3f}")
        failures += 0 if ok else 1
    except Exception as e:
        failures += 1
        check("métrica: literal < basura (distancia)", False, str(e)[:80])

    print(f"\n{'PASS ✓' if failures == 0 else f'FAIL ✗ ({failures} checks)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
