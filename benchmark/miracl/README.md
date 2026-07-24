# MIRACL Korean retrieval benchmark

This manual benchmark exercises AutoRAG's production BM25, MinSync, and hybrid
retrieval paths against the Korean MIRACL development set. It is intentionally
not run in CI: preparation downloads and scans the full Korean corpus, and
MinSync requires a contributor-supplied embedding service.

## Dataset and prerequisites

- [Bun](https://bun.sh/) and this repository's dependencies (`bun install`)
- Network access to Hugging Face for preparation
- Enough local disk for the downloads, normalized data, temporary parsed
  mirrors, and indexes
- For MinSync or hybrid: a MinSync executable plus an explicitly configured,
  reproducible embedder

The benchmark pins the Korean development topics and qrels from
[`miracl/miracl`](https://huggingface.co/datasets/miracl/miracl/tree/5be20db9509754dadad47689368639fcec739c00)
at revision `5be20db9509754dadad47689368639fcec739c00`, and the three
Korean corpus shards from
[`miracl/miracl-corpus`](https://huggingface.co/datasets/miracl/miracl-corpus/tree/d921ec7e349ce0d28daf30b2da9da5ee698bef0d)
at revision `d921ec7e349ce0d28daf30b2da9da5ee698bef0d`. The smoke
profile uses seed `20260723`, 32 queries, and 10,000 deterministic distractors.
The full profile contains all 1,486,752 Korean passages.

MIRACL's topics, qrels, and corpus are licensed under
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Cite the
dataset as:

```bibtex
@article{10.1162/tacl_a_00595,
  author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  title = {{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {11},
  pages = {1114--1131},
  year = {2023},
  month = {09},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00595},
  url = {https://doi.org/10.1162/tacl_a_00595}
}
```

Preparation downloads about 226 MB compressed even for smoke, then streams and
decompresses every corpus shard to select the deterministic subset. Keep
several hundred MB free for smoke. Full preparation, mirror materialization,
and indexing need substantially more space and time; exact requirements depend
on filesystem, network, method, and embedder throughput.

## Smoke benchmark

The CLI requires the output parent directories to exist. Both paths below are
gitignored:

```bash
mkdir -p benchmark/miracl/data benchmark/miracl/runs

bun run benchmark:miracl prepare --profile smoke --output benchmark/miracl/data/smoke
bun run benchmark:miracl run --profile smoke \
  --prepared benchmark/miracl/data/smoke \
  --output benchmark/miracl/runs/bm25-smoke \
  --methods bm25
bun run benchmark:miracl evaluate --run benchmark/miracl/runs/bm25-smoke
```

Each command should exit zero. A valid BM25 smoke run reports 32 evaluated
queries, zero failures, and the four run files described below. Run output
directories must not already exist.

## MinSync and hybrid

Do not substitute a convenient local model when reporting acceptance results.
Use the intended external configuration, or record MinSync and hybrid as not
run. Scores produced by different embedders, model revisions, endpoint kinds,
dimensions, prefixes, or service implementations are not directly comparable.

Create an ignored `benchmark-config.local.json` with real values. This dummy
example is deliberately nonfunctional:

```json
{
  "binaryPath": "/absolute/path/to/minsync",
  "autoInstall": false,
  "embedder": {
    "id": "replace-with-an-immutable-model-revision",
    "baseUrl": "https://embedder.example.invalid/v1",
    "apiKeyEnv": "MIRACL_EMBEDDING_API_KEY",
    "dimension": 1024,
    "queryPrefix": "query: ",
    "passagePrefix": "passage: ",
    "timeoutMs": 60000,
    "batchSize": 64,
    "maxRetries": 3,
    "maxConcurrent": 4
  }
}
```

Set the named environment variable without putting its value in the file or
command history, then run:

```bash
bun run benchmark:miracl run --profile smoke \
  --prepared benchmark/miracl/data/smoke \
  --output benchmark/miracl/runs/all-smoke \
  --methods bm25,minsync,hybrid \
  --config benchmark-config.local.json
bun run benchmark:miracl evaluate --run benchmark/miracl/runs/all-smoke
```

For reproducibility, set `autoInstall` to `false`, retain the exact MinSync
binary and its SHA-256, and use an immutable embedder model/service revision
with identical settings. `autoInstall: true` resolves the latest verified
release and is convenient, but is not a durable version pin. During a run the
benchmark verifies that the executable and MinSync index do not change.
However, reports deliberately disclose only embedder ID, local/remote endpoint
kind, API-key environment-variable name, and dimension. They omit the binary
path/digest, literal base URL, secret value, and operational tuning fields, so
retain those reproducibility details separately without publishing secrets.

## Full benchmark

Full preparation requires the explicit confirmation flag:

```bash
bun run benchmark:miracl prepare --profile full \
  --output benchmark/miracl/data/full \
  --confirm-full
bun run benchmark:miracl run --profile full \
  --prepared benchmark/miracl/data/full \
  --output benchmark/miracl/runs/bm25-full \
  --methods bm25
bun run benchmark:miracl evaluate --run benchmark/miracl/runs/bm25-full
```

To run every method, use a new output directory and the same reviewed MinSync
configuration:

```bash
bun run benchmark:miracl run --profile full \
  --prepared benchmark/miracl/data/full \
  --output benchmark/miracl/runs/all-full \
  --methods bm25,minsync,hybrid \
  --config benchmark-config.local.json
bun run benchmark:miracl evaluate --run benchmark/miracl/runs/all-full
```

## Generated files and schemas

Preparation writes `downloads/topics.tsv`, `downloads/qrels.tsv`, three
compressed corpus shards, normalized `queries.jsonl`, `qrels.jsonl`, and
`corpus.jsonl`, plus `prepared-manifest.json`. The manifest records
`schemaVersion: 1`, `normalizationVersion: 1`, profile, exact source
revisions/URLs, source SHA-256 and byte counts, normalized file names, and
dataset counts. Smoke also records the seed and selected query/document IDs;
full also records normalized-file SHA-256, byte, and record attestations.

Every successful run directory contains exactly:

- `manifest.json`: schema version, profile, dataset counts/revisions and input
  and normalized attestations, embedded evaluation qrels, methods, sanitized
  method configuration, and runtime/commit metadata
- `results.jsonl`: one record per query and method:
  `{schemaVersion, method, queryId, latencyMs, hits, errorCode?}`, where each hit
  is `{documentId, score, rank}`
- `metrics.json`: `{schemaVersion, methods, indexingLatencyMs, peakRssBytes?}`;
  each method has `queryCount`, `failureCount`, `recallAt`, `mrrAt10`,
  `successAt`, `ndcgAt10`, and latency `mean`, `p50`, and `p95`
- `summary.md`: a human-readable rendering of the same dataset,
  configuration, quality, performance, and limitation fields

Published run artifacts contain MIRACL document IDs, not local source paths.
They never contain an API-key value or literal embedder base URL. Review all
artifacts before sharing them; the API-key environment-variable name and
embedder ID are intentionally disclosed for traceability.

## Metrics, timing, and failures

- **nDCG@10** is graded relevance gain through rank 10, normalized by the ideal
  ranking.
- **Recall@5/10/100** is the fraction of positively judged documents retrieved
  by each cutoff.
- **MRR@10** is the reciprocal rank of the first positively judged result
  through rank 10.
- **Success@1/5** is the fraction of queries with at least one positively
  judged result by the cutoff.
- Quality metrics are macro-averaged over query-method records. Unjudged
  documents have zero relevance.
- Query mean, p50, and nearest-rank p95 include successful retrieval calls
  only. Indexing is reported separately; hybrid indexing time is the sum of
  BM25 and MinSync indexing.

A query retrieval failure is persisted as `errorCode: "retrieval-failed"` with
no private exception text. It scores zero in every quality metric, remains in
the query denominator, and is excluded from latency statistics. `run` still
publishes the complete report but exits nonzero if any query failed.
`evaluate` validates and recomputes the saved report, prints `metrics.json` as
JSON, and exits nonzero when any saved query failed. Preparation, indexing,
configuration, integrity, or artifact-validation errors also exit nonzero and
may prevent report publication.

Peak RSS is the maximum reported for the benchmark CLI process as a whole, not
a per-method or per-query measurement. It can include prepared-data loading,
mirror materialization, in-process indexing/retrieval, and report work, and it
does not measure an external MinSync child process's own peak. The field is
omitted when the runtime cannot provide a reliable maximum.

Full MinSync executable and collection-content hashing happens before and
after the query batch, outside reported query latency. Cheap device, inode,
size, and modification-time checks remain at query boundaries. Keep these
integrity checks enabled: although excluded from query latency, full hashing
can warm or evict filesystem cache pages and therefore perturb later retrieval
timings. Compare runs only under equivalent cache, hardware, binary, embedder,
and configuration conditions.
