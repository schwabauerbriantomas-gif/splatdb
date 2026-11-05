#!/usr/bin/env bash
# hermes-hub-ensure: asegura splatsdb del hub de sesiones Hermes en 8872.
# Idempotente: si ya responde, no hace nada. No toca el hub 8871 (m2m legacy).
# Cron sugerido: */10 * * * *  (recuperación ante crash en <=10 min).
set -u
LOG_DIR="/mnt/d/WSL/orch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/hermes-hub.log"
BIN="/mnt/d/splatdb/target/release/splatsdb"
HUB="/mnt/d/splatdb/hermes-hub"
PORT=8872

if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status"'; then
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "$(date -u +%FT%TZ) ERROR: binario splatsdb no encontrado en $BIN" >> "$LOG"
    exit 1
fi

BIN_PATH="$HUB/data/vectors/default.bin"
if [ ! -f "$BIN_PATH" ]; then
    echo "$(date -u +%FT%TZ) skip: sin $BIN_PATH todavía (correr hermes_ingest + hermes_merge)" >> "$LOG"
    exit 1
fi

nohup "$BIN" --dim 768 serve --port "$PORT" --host 127.0.0.1 --data-dir "$HUB" --gpu >> "$LOG" 2>&1 &
echo $! > "$HUB/hub.pid"
echo "$(date -u +%FT%TZ) hub hermes lanzado pid $(cat "$HUB/hub.pid") (puerto $PORT)" >> "$LOG"
