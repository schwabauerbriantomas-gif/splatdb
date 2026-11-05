#!/usr/bin/env bash
# hermes-cycle: ciclo diario completo del hub de sesiones Hermes.
#   ingesta incremental → merge → restart del hub → smoke test
# Cron sugerido: 15 9 * * *  (06:15 Argentina, una vez por día).
# El ensure (*/10) cubre crash-recovery entre ciclos.
set -u
LOG_DIR="/mnt/d/WSL/orch"
HUB="/mnt/d/splatdb/hermes-hub"
SH="/mnt/d/splatdb/semantic-hub"
LOG="$LOG_DIR/hermes-hub.log"
mkdir -p "$LOG_DIR"

say() { echo "$(date -u +%FT%TZ) [cycle] $1" >> "$LOG"; }

say "=== inicio ciclo ==="

# 1. embed server tiene que estar vivo para embeber
bash /mnt/d/embedding-server/start-embed-server.sh >> "$LOG" 2>&1
for i in $(seq 1 90); do
    curl -s --max-time 2 http://127.0.0.1:8860/health | grep -q status && break
    sleep 2
done
if ! curl -s --max-time 2 http://127.0.0.1:8860/health | grep -q status; then
    say "ERROR: embed server no levantó — ciclo abortado"
    exit 1
fi

# 2. ingesta incremental
if ! python3 "$SH/hermes_ingest.py" >> "$LOG" 2>&1; then
    say "ERROR: ingesta falló — ciclo abortado (sin merge ni restart)"
    exit 1
fi

# 3. merge (siempre rebuild del master; es barato y determinista)
if ! python3 "$SH/hermes_merge.py" >> "$LOG" 2>&1; then
    say "ERROR: merge falló — ciclo abortado"
    exit 1
fi

# 4. restart del hub para que cargue el bin nuevo (solo si está vivo y es nuestro)
if [ -f "$HUB/hub.pid" ] && kill -0 "$(cat "$HUB/hub.pid")" 2>/dev/null; then
    say "restart hub (pid $(cat "$HUB/hub.pid"))"
    kill "$(cat "$HUB/hub.pid")" 2>/dev/null
    sleep 2
fi
bash "$SH/hermes-hub-ensure.sh" >> "$LOG" 2>&1
for i in $(seq 1 60); do
    curl -s --max-time 2 http://127.0.0.1:8872/health | grep -q status && break
    sleep 1
done

# 5. smoke: si falla, queda en el log con detalle
if python3 "$SH/hermes_smoke.py" >> "$LOG" 2>&1; then
    say "=== ciclo OK ==="
else
    say "WARN: smoke falló — ver detalles arriba"
    exit 1
fi
