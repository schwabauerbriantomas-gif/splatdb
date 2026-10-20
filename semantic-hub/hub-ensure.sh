#!/usr/bin/env bash
# semantic-hub: asegura SplatsDB en 8871 con el índice semántico (auto-load default.bin).
# Idempotente: si ya responde, no hace nada. Data-dir = raíz del hub (storage agrega data/vectors).
set -u
LOG_DIR="/mnt/d/WSL/orch"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/semantic-hub.log"
BIN="/mnt/d/splatdb/target/release/splatsdb"
HUB="/mnt/d/splatdb/semantic-hub"
PORT=8871

if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"status"'; then
    echo "$(date -u +%FT%TZ) hub ya activo en $PORT" >> "$LOG"
    exit 0
fi

if [ ! -x "$BIN" ]; then
    echo "$(date -u +%FT%TZ) ERROR: binario splatsdb no encontrado en $BIN" >> "$LOG"
    exit 1
fi

cd /mnt/d/splatdb
nohup "$BIN" --dim 768 serve --port "$PORT" --host 127.0.0.1 --data-dir "$HUB" --gpu >> "$LOG" 2>&1 &
echo $! > "$HUB/hub.pid"
echo "$(date -u +%FT%TZ) hub lanzado pid $(cat "$HUB/hub.pid")" >> "$LOG"
