# SplatsDB Documentation Index

| Document | Contents |
|----------|----------|
| [README.md](../README.md) | Project overview, quick start, benchmark summary (honest tables with k + recall per row) |
| [BENCHMARKS.md](../BENCHMARKS.md) | All benchmark tables, each number with source, date, hardware, and reproduction command |
| [CHANGELOG.md](../CHANGELOG.md) | Release changelog |
| [A5_FINDINGS.md](A5_FINDINGS.md) | A5 empirical audit — energy-as-score degrades retrieval; splats validated as routing codebook |

## Benchmark documentation policy

1. Every published QPS number states its **k**, its **recall@k** (or "exact"), its **hardware**, and its **source** (result file or reproduction command).
2. Exact (brute-force) and approximate (ANN) results are never presented in the same comparison row or implied to be equivalent.
3. Numbers without a preserved source are **removed**, not restated; removals are recorded in BENCHMARKS.md → "Removed claims".
4. Benchmark chart images are renderings of the tables, not sources of truth; if table and PNG disagree, the table wins and the PNG is regenerated.
