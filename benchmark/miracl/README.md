# MIRACL Korean retrieval benchmark

This manual benchmark exercises AutoRAG's production retrieval semantics
against the Korean MIRACL development set. Smoke runs support BM25, MinSync,
and hybrid retrieval. Full runs support streaming Tantivy BM25 only; production
MinSync requires one file per passage, so this benchmark does not claim a
scalable 1,486,752-file MinSync or hybrid workflow. It is intentionally not run
in CI: preparation downloads and scans the full Korean corpus, and MinSync
requires a contributor-supplied embedding service.

## Dataset and prerequisites

- [Bun](https://bun.sh/) and this repository's dependencies (`bun install`)
- Network access to Hugging Face for preparation
- Enough local disk for the downloads, normalized data, and indexes
- For smoke MinSync or hybrid: an explicitly pinned MinSync executable plus an
  explicitly configured, reproducible embedder

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
decompresses every corpus shard to select the deterministic subset. Each
download has a ten-minute timeout, and source and normalized inputs are parsed
with byte and record bounds. Keep several hundred MB free for smoke. Full
preparation and Tantivy indexing need substantially more space and time; exact
requirements depend on filesystem, network, and storage throughput. Full
indexing streams the attested `corpus.jsonl` directly into Tantivy and does not
materialize one mirror file per passage or retain the corpus as an in-memory
array.

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

The benchmark rejects omitted or ambiguous pinning: `binaryPath`,
`autoInstall: false`, `embedder.id`, `embedder.dimension`, and
`embedder.timeoutMs` are required, and no executable is resolved from `PATH`,
an installer, or a cache. MinSync init, sync, and query subprocesses use the
configured timeout and cap captured stdout at 16 MiB and stderr at 1 MiB.
During a run the benchmark verifies that the executable and MinSync index do
not change.

Reports record the executable SHA-256, immutable model ID, local/remote
endpoint kind, authentication kind, dimension, timeout and process-output
caps, SHA-256 hashes of both prefixes, and supplied batch/retry/concurrency
settings. They omit the binary path, literal base URL, API-key environment
variable name, and secret value.

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

Full MinSync and hybrid commands are rejected before prepared data is loaded.
The full BM25 path uses the production markdown normalization, BM25 chunking,
query parsing, virtual-path identity, and document-level ranking semantics. It
requires Tantivy rather than silently falling back to an in-memory
TypeScript-wide-corpus index.

## Generated files and schemas

Preparation writes `downloads/topics.tsv`, `downloads/qrels.tsv`, three
compressed corpus shards, normalized `queries.jsonl`, `qrels.jsonl`, and
`corpus.jsonl`, plus `prepared-manifest.json`. The manifest records
`schemaVersion: 1`, `normalizationVersion: 1`, profile, exact source
revisions/URLs, source SHA-256 and byte counts, normalized file names, and
dataset counts. Every profile records SHA-256, byte, and record attestations
for normalized queries, qrels, and corpus. Smoke also records the seed and
selected query/document IDs. A run recomputes and compares all three
attestations before indexing; matching IDs or counts alone are insufficient.

Every successful run directory contains exactly:

- `manifest.json`: schema version, profile, dataset counts/revisions and input
  and normalized attestations, embedded evaluation qrels, methods, sanitized
  method configuration, actual BM25 engine, and runtime/commit metadata
- `results.jsonl`: one record per query and method:
  `{schemaVersion, method, queryId, latencyMs, hits, errorCode?}`, where each hit
  is `{documentId, score, rank}`
- `metrics.json`:
  `{schemaVersion, methods, indexingLatencyMs, peakRssBeforeReportBytes?}`;
  each method has `queryCount`, `failureCount`, `recallAt`, `mrrAt10`,
  `successAt`, `ndcgAt10`, and latency `mean`, `p50`, and `p95`
- `summary.md`: a human-readable rendering of the same dataset,
  configuration, quality, performance, and limitation fields

Published run artifacts contain MIRACL document IDs, not local source paths.
They never contain an API-key value, API-key environment-variable name, literal
embedder base URL, or executable path. Review all artifacts before sharing
them; the immutable embedder ID and executable digest are intentionally
disclosed for traceability.

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
- Retrieval initially asks for 100 chunks and doubles the bounded request up to
  1,600 when chunk deduplication has not yet produced 100 unique MIRACL
  documents. Document scores use the best matching chunk.
- Query mean, p50, and nearest-rank p95 include successful retrieval calls
  only. Indexing is reported separately; hybrid indexing time is the sum of
  BM25 and MinSync indexing.

A query retrieval failure is persisted as `errorCode: "retrieval-failed"` with
no private exception text. It scores zero in every quality metric, remains in
the query denominator, and is excluded from latency statistics. `run` still
publishes the complete report but exits nonzero if any query failed.
`evaluate` requires exactly `manifest.json`, `results.jsonl`, `metrics.json`,
and `summary.md`; it rejects extra or missing files, validates that
`summary.md` is the canonical rendering of the structured artifacts,
recomputes all metrics, prints `metrics.json` as JSON, and exits nonzero when
any saved query failed. Preparation, indexing, configuration, integrity, or
artifact-validation errors also exit nonzero and may prevent report
publication.

`peakRssBeforeReportBytes` is the maximum reported for the benchmark CLI
process as a whole through prepared-data loading, indexing, retrieval,
evaluation, and report-input construction. It is sampled before report
serialization, staging, and publication, and it does not measure an external
MinSync child process's own peak. It is not a per-method or per-query
measurement. The field is omitted when the runtime cannot provide a reliable
maximum.

MinSync executable and collection-content hashing happen before and after the
smoke query batch, outside reported query latency. The source-path map is built
once after sync, while cheap device, inode, size, and modification-time checks
remain at query boundaries. Keep these integrity checks enabled: although
excluded from query latency, content hashing can warm or evict filesystem cache
pages and therefore perturb later retrieval timings. Compare runs only under
equivalent cache, hardware, binary, embedder, and configuration conditions.
