# Hermes Memory Hub — recall semántico de sesiones

Capa de memoria semántica para el agente Hermes: indexa los mensajes de
conversación de `state.db` en un SplatsDB dedicado y expone recall por
**significado** (no por keyword). Complementa el FTS5 nativo de Hermes,
que gana en identificadores literales pero falla cuando la pregunta usa
otras palabras que la sesión original.

## Arquitectura

```
Hermes state.db (SQLite, SOLO LECTURA)
        │  hermes_ingest.py   (incremental, dedup por hash)
        ▼
hermes-hub/parts/hermes.db ──► hermes_merge.py ──► hermes-hub/data/vectors/default.bin
        (part: text+vec+meta)         (master sidecar + shard bin)   + hermes-hub/sidecar.db
                                                                            │
embeddinggemma-2 (:8860, GPU 3090, 768d)          splatsdb serve (:8872) ◄──┘
        ▲                                                  │
        └──── hermes_recall.py (embed query → search → join sidecar)

14 repos locales (código) ──► ingest.py --hub hermes-hub-code ──► hermes-hub-code/
                                                                      └─► splatsdb (:8873)
```

- **8871**: hub existente con corpus m2m legacy (26.273 chunks, oct-2026) — no se toca.
- **8872**: sesiones Hermes (hub puro), data-dir `hermes-hub/`.
- **8873**: código de 14 repos (~6.5k chunks), data-dir `hermes-hub-code/` — hub
  puro a propósito: embeddinggemma es sesgado a prosa y en un hub mixto los
  mensajes que *discuten* código le ganan al código mismo (code hits ~rank 25
  en top-30; en el hub puro, top-3).
- El estado vivo de Hermes **nunca se escribe**: la fuente se abre en modo RO.
- El sidecar (`sidecar.db`) es el JOIN post-búsqueda: el server devuelve
  índices; el texto y la metadata (`{message_id, session_id, role, title, date}`)
  viven client-side. `chunks.rowid = vector_index + 1`.
- Orden de merge: `hermes` primero — preserva los vector_idx de sesiones
  cuando se agregan parts nuevos.

## Filtros de ingesta

Entra un mensaje si: `role ∈ {user, assistant}` AND `active=1` AND
`length ≥ 60` AND no empieza con notificaciones de sistema
(`[IMPORTANT:`, `[ASYNC DELEGATION`, `[OUT-OF-BAND`, `[Background process`,
`[Command timed out`) AND no es blob JSON puro (`{` o `[` inicial).
Texto truncado a 1500 chars (consistente con el benchmark).
Dedup por blake2b-96 del texto: re-envíos idénticos no duplican vectores.

## Embeddings

- Modelo: `google/embeddinggemma-2` local (`/mnt/d/embedding-server`, venv
  `~/.venvs/embed-server`, puerto 8860, MRL truncado a 768).
- **Asimetría de prompts (crítico para la calidad medida)**: documentos sin
  prompt; queries con `prompt_name=Retrieval-query`. El smoke test de
  `faststart.sh` del hub 8871 embebe queries sin el prompt — por eso sus
  scores de demo salían degradados (~0.35) frente a los de este pipeline (~0.6).

## Números medidos (2026-11-05, corpus real)

Benchmark de falsificación (`cache/scratch/mem-bench/bench.py` del perfil
Hermes; corpus: 4.411 mensajes muestreados de state.db):

| Métrica | FTS5 (actual) | Vector (este hub) |
|---|---|---|
| Recall@10 sesión, queries paráfrasis (n=8) | 2/8 | **7/8** |
| MRR sesión | 0.27 | **0.60** |
| Recall@10 sesión, queries literales (n=3) | 3/3 | 3/3 |
| Latencia query | ~14 ms | ~115–220 ms end-to-end |

En 5 de 8 queries paráfrasis el FTS devolvió cero resultados relevantes y el
vector trajo la sesión correcta en rank 1–7. Advertencias honestas: 11 queries
armadas a mano por el agente (no es un dataset público), un solo corpus, sin
cross-validación; la magnitud del gap sostiene la dirección del resultado, no
el dígito exacto. Queries basura: FTS devuelve 105–11.594 matches irrelevantes
(sin rechazo); el vector ordena bien pero las similitudes absolutas son altas
incluso para basura (0.59–0.70) → sirve como top-k para leer, no como umbral
automático de rechazo.

Operación real: primera ingesta 5.585 docs en 81 s (~69 msg/s), bin 17.2 MB.
Smoke post-restart: 3 casos de paráfrasis en rank 5/2/1.

## Semántica del score (importante)

El `score` que devuelve el server es una **DISTANCIA: menor = más cerca**.
El server ya ordena best-first (ascendente); no re-invertir en el cliente.
Calibración medida sobre este corpus:

| tipo de query | sesiones (8872) | código (8873) |
| --- | --- | --- |
| frase literal de la fuente | 0.42–0.51 | — |
| paráfrasis / tema del corpus | 0.53–0.60 | 0.67–0.70 |
| pregunta válida fuera del corpus | — | 0.70–0.76 |
| basura sin relación | 0.77+ | 0.81+ |

Corte de relevancia: `--max-dist 0.60` (sesiones) / `≤ 0.70` (código).
El smoke incluye un check de sanidad (`literal < basura`) que detecta
inversiones de esta semántica. Lección: la hipótesis "no se puede rechazar
basura por score" del benchmark inicial era un artefacto de interpretar el
score como similitud — con la semántica correcta la separación es limpia.

## Operación (runbook)

```bash
# ciclo completo diario (sesiones + re-ingesta de código + merges + restarts + smoke)
bash /mnt/d/splatdb/semantic-hub/hermes-cycle.sh

# ensure idempotente AMBOS hubs (crash-recovery; cron cada 10 min)
bash /mnt/d/splatdb/semantic-hub/hermes-hub-ensure.sh

# smoke end-to-end (health x3 + consistencia x2 + 3 recalls sesión + 1 de código)
python3 /mnt/d/splatdb/semantic-hub/hermes_smoke.py

# ingesta manual / rebuild desde cero
python3 /mnt/d/splatdb/semantic-hub/hermes_ingest.py [--rebuild]
python3 /mnt/d/splatdb/semantic-hub/hermes_merge.py --parts hermes

# código: re-ingesta de los 14 repos + merge del hub de código
python3 /mnt/d/splatdb/semantic-hub/ingest.py --hub /mnt/d/splatdb/hermes-hub-code \
  --name repos --kind repo --cap 500 --repo /mnt/d/splatdb --repo /mnt/d/vuupta ...
python3 /mnt/d/splatdb/semantic-hub/hermes_merge.py --hub /mnt/d/splatdb/hermes-hub-code --parts repos
```

Cron del usuario (patrón existente de la máquina — sin sudo, sin systemd):

```
*/10 * * * * bash /mnt/d/splatdb/semantic-hub/hermes-hub-ensure.sh
15 9 * * *   bash /mnt/d/splatdb/semantic-hub/hermes-cycle.sh
```

Logs: `/mnt/d/WSL/orch/hermes-hub.log`.

### Lección operativa que motivó el watchdog

El server de embeddings estuvo caído 10 días (26-oct → 5-nov-2026) por un
venv roto mientras un trigger diario lo intentaba en silencio: el fallo solo
dejaba una línea en el log y ningún humano la leía. El ensure cada 10 min +
el ciclo diario con smoke que escribe PASS/FAIL al log existen para que eso
no se repita. Si algo falla, `hermes-cycle.sh` aborta **antes** del merge —
el hub sigue sirviendo el índice de ayer en vez de romperse.

## Gate de decisión (KEV / System One)

**Estado: NO shippable como filtro — server disponible para experimentos.**

Calibración con 48 mensajes reales del usuario (state.db): las distribuciones
de p(sí) se solapan — basura real 0.09-0.41 (mediana 0.23), mensajes de
proyecto 0.10-0.80 (mediana 0.38). Ningún threshold filtra basura sin
perder reales (th=0.12 filtra ~20% de la basura y pierde reales; th=0.40
filtra 18/21 pero pierde 14/27). La mitad de la "basura" son comandos
("Apaga la PC") con etiqueta genuinamente ambigua para la pregunta `noul
"¿recuperar memoria mejora la respuesta?"`. El 8/8 inicial fue overfit a
extremos elegidos a mano. El hook NO consulta el gate (fail-open total);
los tiers de distancia + FTS5 siguen siendo el único criterio.

Infraestructura desmontada del stack (9b61d0d, a pedido del usuario tras
la calibración): servicio muerto, autostart fuera del ensure, pidfile
borrada. El clon /mnt/d/kev y los caches HF quedan en disco. Para
reactivar: `cd /mnt/d/kev && HF_HOME=/mnt/d/huggingface
KEV_CUDA_GRAPHS=0 KEV_PREFIX_CACHE=0 KEV_DEVICE_PROBE_TIMEOUT=180
nohup uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b
--port 8008 --device cuda &` (requiere `flash-linear-attention` en el
venv — sin él el load no termina). Set de calibración en scratch
(`kev_calib_set.json`). El 0.8B no sirve ni como experimento (1/4).
Camino futuro: reformular la pregunta (comandos vs preguntas),
recolectar p en log-only, o Clef-Flash 9B si hay VRAM.

## Limitaciones conocidas

- Mensajes con contenido thin (blobs JSON de health-checks) no son
  recuperables por significado — caso fallado del benchmark.
- La ingesta es append-only: un mensaje borrado/editado en state.db no
  desaparece del índice hasta el siguiente `--rebuild`.
- El restart diario del hub (~2 s) corta búsquedas en vuelo: está agendado
  a las 06:15 Argentina a propósito.
- Similitudes absolutas altas en embeddinggemma: no usar `--min-score` como
  filtro de calidad sin calibrar antes contra casos propios.

## Privacidad

El corpus son conversaciones privadas: vive en `hermes-hub/` (gitignored,
nunca commitea) y en la máquina local. Los tests usan datos sintéticos.
El CI no requiere GPU ni servicios: los tests mockean el server de embeddings.
