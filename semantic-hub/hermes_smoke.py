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
     "20261026_191031", 5),
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

    print(f"\n{'PASS ✓' if failures == 0 else f'FAIL ✗ ({failures} checks)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
