# A5 — Empirical audit: energy-as-score vs cosine retrieval (2026-09-22)

Causal experiment: does splat energy `E(x) = -log(Σ αᵢ·exp(-κᵢ·‖x−μᵢ‖²))` improve
retrieval when mixed into the ranking score? Short answer: **no — it degrades it**,
consistently, on two corpora, with hand-written semantic ground truth and bootstrap
confidence intervals. Splats remain valuable as a **compression/routing codebook**.

## Setup

- Embeddings: `all-MiniLM-L6-v2` (384-d, normalized), same model for corpus+queries.
- Corpus A (local-technical): 9,533 chunks from real local docs (skills .md, clone
  READMEs, Rust doc-comments, git RelNotes). 100 queries: 50 verbatim-first-sentence,
  50 mechanical paraphrase.
- Corpus B (diverse): 20 Newsgroups, 10,000 chunks, 20 balanced topics.
  60 queries: semantic-intent paraphrases **hand-written by the auditor** (zero lexical
  overlap with the target chunk).
- Conditions: pure cosine (baseline) vs energy fusion at several weights, splat
  configurations k∈{50,200,1000}, κ∈{2,8,32}, per-cluster κ, query-conditioned E(x|q),
  true kNN density (no splat approximation error), Boltzmann sampling coverage.
- KMeans seeds {0,1,2}; noise floor ±5 pts (n=100/60) declared a priori; bootstrap
  10k resamples for CIs. Scripts + JSONs: `swarm10/energy_eval/` (local workspace).

## Headline results

| Condition | Corpus A R@10 | Corpus B R@10 | Verdict |
|---|---|---|---|
| Pure cosine (baseline) | 0.90 | 0.90 | — |
| Global fusion 0.5·cos + 0.5·energy | 0.00–0.04 | — | total collapse |
| Pool-fusion w=0.25 (steelman) | 0.62 | 0.65 | **−25 pts, CI95 [−0.38,−0.12], significant** |
| Per-cluster κ (κᵢ=1/2σᵢ²) | 0.64 | 0.70 | still significant damage |
| True kNN density, w=0.1 | 0.89 | 0.883 | neutral at best (CI touches 0 from below) |
| Query-conditioned E(x\|q) | 0.61 | 0.80 | best energy variant; still loses to cosine |
| Boltzmann p(GT\|top-100) | ≤0.0092 (uniform 0.01) | — | energy ranks GT below random |

## Binary route audit (v2.5.0)

- `index` + `search-file` on raw embeddings: **exact** — pooled R@10 0.90±0.06 (100q),
  top-10 overlap 1.0 vs brute-force. (This route has no energy in the score; α/κ in the
  output are decorative there.)
- Semantic-memory route (`serve` + POST /store + /search, where splats participate in
  scoring): R@10 **0.18** (−72 pts) — independently consistent with the LongMemEval
  native-vs-brute-force gap (0.304 vs 0.966) reported in the benchmark audit.
- Bugs found (tracked in issues #1–#5): `ingest` success+empty store; `status` panic at
  384-dim; `serve` ignores `--dim` (hardcoded 640); `search -q` rejects negative CSV
  components; `search-file` missing `--shard`.

## What IS validated

- **Splats as compression codebook / routing structure** (IVF-style): probe top-5
  splats → exact re-score inside → R@10 0.66 reading only 3.4% of the corpus;
  k=1000 → 0.72 @ 1.4%. Scaling codebook to ~√N is the right regime.
- **Exact CLI route** (`index`/`search-file`) — production-safe today.
- Confidence *display* from κ (flagging OOD queries) is orthogonal to this audit and
  remains plausible; what is falsified is energy/density as a *ranking component*.

## Recommendation

Score with cosine (or HNSW on raw vectors). Use splats for storage compression and
candidate routing. Do not mix E(x) or density into the ranking score; if experimenting
further, the only direction not falsified is fully query-conditioned energy with
learned per-cluster parameters — and its measured ceiling so far is below cosine.

*Full data: `swarm10/energy_eval/results_*.json` (local), report `swarm10/results/A5.json`.*
