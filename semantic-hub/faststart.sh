#!/usr/bin/env bash
# faststart.sh — pipeline completo splatDB + embeddinggemma-2 en un comando.
# 1) embedding server (:8860, google/embeddinggemma-2 vía SentenceTransformer)
# 2) hub splatDB (:8871, búsqueda GPU exacta)
# 3) smoke end-to-end
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EMB_DIR="/mnt/d/embedding-server"
LOG="/mnt/d/WSL/orch"
mkdir -p "$LOG"

wait_http() { # url, timeout_s
  local url="$1" t="${2:-60}"
  for _ in $(seq 1 "$t"); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

echo "── 1/3 embedding server (google/embeddinggemma-2, :8860)"
if curl -sf http://127.0.0.1:8860/health >/dev/null 2>&1; then
  echo "   ya activo"
else
  bash "$EMB_DIR/start-embed-server.sh" || true
  wait_http http://127.0.0.1:8860/health 180 \
    || { echo "   ERROR: embedding server no levantó (ver $LOG/embedding-server.log)"; exit 1; }
  echo "   listo"
fi

echo "── 2/3 hub splatDB (:8871, GPU exacto)"
bash "$ROOT/semantic-hub/hub-ensure.sh" >/dev/null 2>&1 || true
wait_http http://127.0.0.1:8871/health 120 \
  || { echo "   ERROR: hub no levantó (ver $LOG/semantic-hub.log)"; exit 1; }
echo "   $(curl -s http://127.0.0.1:8871/health)"

echo "── 3/3 smoke end-to-end"
EMB=$(curl -s -X POST http://127.0.0.1:8860/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": "licitaciones de obra pública en Mendoza", "model": "embeddinggemma-2"}' \
  | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['data'][0]['embedding']))")
curl -s -X POST http://127.0.0.1:8871/search \
  -H 'Content-Type: application/json' \
  -d "{\"query\": \"licitaciones\", \"embedding\": $EMB, \"top_k\": 3}" \
| python3 -c "
import json, sys
r = json.load(sys.stdin)
print('   backend:', r.get('backend', 'cpu'))
for x in r['results'][:3]:
    print(f\"   [{x['index']}] score={x['score']:.4f}\", x.get('topic', ''), x.get('confidence', ''))
"

cat <<'EOF'

✓ Pipeline arriba:
    embeddings  → POST http://127.0.0.1:8860/v1/embeddings  (google/embeddinggemma-2, 768d)
    búsqueda    → POST http://127.0.0.1:8871/search         {"query", "embedding", "top_k"}
    campo exacto→ POST http://127.0.0.1:8871/field-search   {"embedding", "top_k"}
                  {"mode": "two-phase", "nprobe": 32} para modo IVF aproximado
EOF
