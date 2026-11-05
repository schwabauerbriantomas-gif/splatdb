#!/usr/bin/env bash
# hermes-hub-ensure: asegura los hubs splatsdb de memoria Hermes.
#   :8872 sesiones (hermes-hub)  |  :8873 código (hermes-hub-code)
# Idempotente POR HUB (sin early-exit: un hub vivo no debe saltear el otro).
# No toca el 8871 (memorias m2m legacy, otro pipeline).
# Cron sugerido: */10 * * * *  (recuperación ante crash en <=10 min).
set -u
LOG_DIR="/mnt/d/WSL/orch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/hermes-hub.log"
BIN="/mnt/d/splatdb/target/release/splatsdb"

ensure_hub() { # puerto, data-dir, etiqueta
    local PORT="$1" HUB="$2" LABEL="$3"
    if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status"'; then
        return 0
    fi
    if [ ! -x "$BIN" ]; then
        echo "$(date -u +%FT%TZ) ERROR: binario splatsdb no encontrado en $BIN" >> "$LOG"
        return 1
    fi
    if [ ! -f "$HUB/data/vectors/default.bin" ]; then
        echo "$(date -u +%FT%TZ) skip $LABEL: sin $HUB/data/vectors/default.bin todavía (correr ingest+merge)" >> "$LOG"
        return 1
    fi
    nohup "$BIN" --dim 768 serve --port "$PORT" --host 127.0.0.1 --data-dir "$HUB" --gpu >> "$LOG" 2>&1 &
    echo $! > "$HUB/hub.pid"
    echo "$(date -u +%FT%TZ) hub $LABEL lanzado pid $(cat "$HUB/hub.pid") (puerto $PORT)" >> "$LOG"
}

ensure_hub 8872 "/mnt/d/splatdb/hermes-hub" "sesiones"
ensure_hub 8873 "/mnt/d/splatdb/hermes-hub-code" "código"

# gate KEV (System One, :8008): clasificador binario calibrado del hook de
# memoria. Servido con uv desde /mnt/d/kev; ~2GB VRAM (0.8B bf16).
KEV_LOG="$LOG_DIR/kev-serve.log"
KEV_PID="/mnt/d/kev/kev-serve.pid"
if ! curl -s --max-time 3 http://127.0.0.1:8008/v1/models | grep -q 'kev'; then
    # si ya hay una instancia cargando (pid viva), no duplicar
    if [ -f "$KEV_PID" ] && kill -0 "$(cat "$KEV_PID")" 2>/dev/null; then
        echo "$(date -u +%FT%TZ) kev-serve pid $(cat "$KEV_PID") sigue cargando — no relanzo" >> "$LOG"
    elif [ -x "$HOME/.local/bin/uv" ] && [ -f /mnt/d/kev/kev/serve.py ]; then
        HF_HOME=/mnt/d/huggingface KEV_CUDA_GRAPHS=0 KEV_PREFIX_CACHE=0 \
        KEV_DEVICE_PROBE_TIMEOUT=180 nohup "$HOME/.local/bin/uv" run --extra serve --project /mnt/d/kev \
            python -m kev.serve --run jaredpalmer/kev-4b --port 8008 \
            >> "$KEV_LOG" 2>&1 &
        echo $! > "$KEV_PID"
        echo "$(date -u +%FT%TZ) kev-serve lanzado pid $(cat "$KEV_PID") (:8008)" >> "$LOG"
    fi
fi
exit 0
