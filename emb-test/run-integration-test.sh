#!/usr/bin/env bash
# Test de integración: embeddinggemma-2 (omnimodal) → SplatsDB API
# Requisitos:
#   - embeddings: ~/.venvs/embed-server/bin/python + server 8860 (bash /mnt/d/embedding-server/start-embed-server.sh)
#   - splatsdb:   cd /mnt/d/splatdb && ./target/release/splatsdb --dim 768 serve --port 8870 \
#                   --host 127.0.0.1 --data-dir /mnt/d/splatdb/emb-test/data
#     OJO: --dim va ANTES de serve (flag global, default 64). Sin él: "expected 64, got 768".
# Resultado esperado (medido 2026-10-20): texto→texto 6/6 top-1; cross-modal: la imagen
# correcta es la mejor imagen del ranking 3/3 (los textos del mismo tema pueden superar
# a imágenes sintéticas — ver README del test).
set -u
EMB=http://127.0.0.1:8860/v1/embeddings
SPLAT=http://127.0.0.1:8870
PY=~/.venvs/embed-server/bin/python

curl -s --max-time 5 http://127.0.0.1:8860/health >/dev/null && echo "embeddings 8860: OK" || { echo "embeddings 8860: MUERTO"; exit 1; }
curl -s --max-time 5 $SPLAT/health >/dev/null && echo "splatsdb 8870: OK" || { echo "splatsdb 8870: MUERTO"; exit 1; }

$PY - <<'EOF'
import json, urllib.request
import numpy as np

EMB = "http://127.0.0.1:8860/v1/embeddings"
SPLAT = "http://127.0.0.1:8870"

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

def embed_texts(texts):
    return [x["embedding"] for x in post(EMB, {"input": texts})["data"]]

docs = [
    ("pcb-1",  "El diseño de la placa HPA usa membrana FKM y piloto neumático 3V1"),
    ("pcb-2",  "KiCad permite rutear pistas de 0.2mm y verificar la direccionalidad de conectores antes de rotar"),
    ("cocina-1", "La torta de manzana lleva canela, azúcar mascabo y masa quebrada"),
    ("cocina-2", "El asado criollo se cocina a las brasas con leña y sal gruesa"),
    ("futbol-1", "Boca ganó el superclásico en La Bombonera con gol de Schott"),
    ("futbol-2", "Messi convirtió un golazo de tiro libre en la última fecha"),
    ("inmob-1", "El departamento de 3 ambientes en Recoleta se vende a 150 mil dólares"),
    ("inmob-2", "El monoambiente en Palermo está alquilado a 350 mil pesos por mes"),
    ("rust-1", "SplatsDB combina HNSW con knowledge graph y devuelve confidence kappa"),
    ("rust-2", "El build release de Rust en NTFS vía WSL tardó tres minutos con diez segundos"),
    ("viaje-1", "Bariloche tiene nieve y skiing en invierno en el Cerro Catedral"),
    ("viaje-2", "El tren a las nubes cruza la cordillera de los Andes a 4000 metros"),
]

# --- ojo: SplatsDB /search devuelve "index" (orden de inserción), no el id ---
order = [d[0] for d in docs]  # se extiende con las imágenes después
print(f"store de {len(docs)} textos con embedding real 768d...")
for id_, text in docs:
    v = embed_texts([text])[0]
    post(SPLAT + "/store", {"id": id_, "text": text, "embedding": v, "category": "embtest"})
print("  ok")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("google/embeddinggemma-2", device="cuda")
print("store de 3 imágenes (multimodal)...")
for name in ["pcb", "comida", "paisaje"]:
    v = np.asarray(m.encode({"image": [f"/mnt/d/splatdb/emb-test/{name}.png"]})).reshape(-1)
    v = (v / np.linalg.norm(v)).tolist()
    post(SPLAT + "/store", {"id": f"img-{name}", "text": f"[imagen] {name}", "embedding": v})
    order.append(f"img-{name}")
print("  ok")

def search_rank(q, k=5):
    emb = embed_texts([q])[0]
    r = post(SPLAT + "/search", {"query": q, "embedding": emb, "top_k": k})
    ranked = sorted(r["results"], key=lambda x: x["score"])  # score = distancia, menor mejor
    return [(order[x["index"]], round(x["score"], 3)) for x in ranked]

print("\n=== texto→texto (top-1 del tema correcto) ===")
tq = [("cómo rutear una placa de circuito impreso", "pcb"),
      ("receta para cocinar un postre rico", "cocina"),
      ("resultado del partido de fútbol", "futbol"),
      ("alquilar vivienda en Buenos Aires", "inmob"),
      ("memoria vectorial para agentes de IA", "rust"),
      ("esquiar en la nieve en invierno", "viaje")]
h = 0
for q, t in tq:
    r = search_rank(q)
    ok = r[0][0].split("-")[0] == t
    h += ok
    print(f"  {'OK ' if ok else 'MAL'} {q[:44]:46} top1={r[0][0]}")
print(f"  => {h}/{len(tq)}")

print("\n=== texto→imagen (cross-modal; la imagen correcta debe ser la mejor imagen) ===")
xq = [("una placa de circuito impreso verde con pistas doradas", "pcb"),
      ("un plato de comida con verduras y tomate", "comida"),
      ("montañas con cielo azul y sol", "paisaje")]
h = 0
for q, t in xq:
    r = search_rank(q, k=15)
    imgs_rank = [(i, pos+1) for pos, (i, _) in enumerate(r) if i.startswith("img-")]
    best_img = imgs_rank[0][0] if imgs_rank else None
    ok = best_img == f"img-{t}"
    h += ok
    print(f"  {'OK ' if ok else 'MAL'} {q[:44]:46} mejor imagen: {best_img} (pos {imgs_rank[0][1]}/15)")
print(f"  => {h}/{len(xq)}")
EOF