#!/usr/bin/env python3
"""Paso 1 del bench latente: query set + ground truth brute-force.

Genera 30 queries (mix repo/memoria, es+en), embeds via 8860, calcula el
top-10 exacto por L2 sobre los 26.273 vectores crudos del hub.
Salida: bench-latent/queries.json [{q, vec, gt:[10 idx]}...]
"""
import json
import struct
import time
import urllib.request
import numpy as np

HUB = "/mnt/d/splatdb/semantic-hub"
OUT = f"{HUB}/bench-latent"

QUERIES = [
    ("scraper que colecta licitaciones públicas de Argentina", "repo"),
    ("custom CUDA kernel for vector similarity search", "repo"),
    ("diseño del pulsador del pulverizador HPA con membrana", "repo"),
    ("autenticación con google service account", "repo"),
    ("migración de memoria sqlite a embeddings", "repo"),
    ("componente react para dashboard con gráficos", "repo"),
    ("configuración de cron jobs y monitoreo", "repo"),
    ("esquema de base de datos para alertas", "repo"),
    ("test end-to-end con playwright", "repo"),
    ("render 3D gaussian splatting pipeline", "repo"),
    ("kicad pcb footprint library python", "repo"),
    ("fastapi endpoint with pydantic validation", "repo"),
    ("YOLO training dataset augmentation", "repo"),
    ("docker compose service healthcheck", "repo"),
    ("git workflow rebase conflict resolution", "repo"),
    ("agenda semanal con prioridades y planificación", "memoria"),
    ("notas del día sobre debugging de sesión", "memoria"),
    ("recordatorio de compromiso pendiente con el usuario", "memoria"),
    ("elección de modelo según costo y complejidad", "memoria"),
    ("skills con dependencias rotas y contexto saturado", "memoria"),
    ("investigación de mercado de empleo remoto", "memoria"),
    ("resumen de conversación sobre despliegue", "memoria"),
    ("lecciones aprendidas de un error de build", "memoria"),
    ("plan de trabajo para la semana", "memoria"),
    ("métricas de performance de tokens y costos", "memoria"),
    ("quantization of GGUF models for inference", "repo"),
    ("websocket server for realtime updates", "repo"),
    ("validación de esquemas JSON en rust", "repo"),
    ("sistema de backup y restore incremental", "repo"),
    ("búsqueda semántica con HNSW en GPU", "repo"),
]


def embed(texts):
    req = urllib.request.Request(
        "http://127.0.0.1:8860/v1/embeddings",
        data=json.dumps({"input": texts}).encode(),
        headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return [x["embedding"] for x in d["data"]]


def main():
    import os
    os.makedirs(OUT, exist_ok=True)

    vecs_file = f"{HUB}/data/vectors/default.bin"
    with open(vecs_file, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = struct.unpack("<Q", f.read(8))[0]
        X = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    print(f"base: {n} x {d}")

    vecs = embed([q for q, _ in QUERIES])
    Q = np.array(vecs, dtype=np.float32)
    gt_all = []
    t0 = time.monotonic()
    for i in range(len(QUERIES)):
        d2 = np.sum((X - Q[i]) ** 2, axis=1)
        gt = np.argsort(d2)[:10].tolist()
        gt_all.append(gt)
    bt = time.monotonic() - t0
    print(f"brute force 30 queries: {bt:.2f}s ({bt/30*1000:.0f} ms/query)")

    out = [{"q": q, "kind": k, "vec": v, "gt": g}
           for (q, k), v, g in zip(QUERIES, vecs, gt_all)]
    with open(f"{OUT}/queries.json", "w") as f:
        json.dump(out, f)
    kinds = {}
    for (q, k), g in zip(QUERIES, gt_all):
        kinds.setdefault(k, []).append(g[0] // 1000)
    print("GT sample por kind (idx//1000):",
          {k: v[:5] for k, v in kinds.items()})
    print(f"OK → {OUT}/queries.json")


if __name__ == "__main__":
    main()
