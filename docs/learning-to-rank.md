# Learning-to-rank training data

AutoRAG fuses candidates from several retrieval channels: MinSync dense vectors and SPLADE sparse terms over local
files, every Jev-selected datasource, the web, and the tools the model calls during verification. Today the fusion is
hand-tuned (min-max normalization, a capped corroboration bonus, an optional reranker). Every search already shows
which candidates mattered: the final answer cites them, and Jev judges whether a citation is relevant and whether it
really supports the answer. This page describes the log that keeps those signals in a form a learning-to-rank (LTR)
model such as LightGBM LambdaRank can be trained on.

This PR only collects and exports the data. No model is trained or served, and nothing in the search path changes
order because of it.

## Why retrieval memory alone is not training data

Retrieval memory (`memory.json`, see [retrieval-memory.md](retrieval-memory.md)) keeps only evidence Jev judged to
support an answer (P(supports) >= 0.7). Those are positives with no record of what they beat. A ranker learns by
comparing candidates inside one query, so it needs:

| Need | Why | Where memory falls short |
|------|-----|--------------------------|
| Query groups | LambdaRank optimizes order within one query | memory has no notion of the candidate pool |
| Negatives | a model that never sees what lost cannot learn to demote it | only the winners are stored |
| Serve-time features | features recomputed later describe an index, corpus, and memory that have moved on | memory stores the excerpt, not the channel ranks and scores |
| Graded labels | "cited and supported" is stronger evidence than "cited", which is stronger than "never cited" | memory stores only one grade, above a cut |
| Exposure | a candidate the model never saw cannot be cited; that is position bias, not irrelevance | memory records nothing about what was shown |

## What is logged

Each finished search run appends one record to `<memoryPath>.ranking.db` (SQLite, beside `memory.json`; one row per
run, so independent processes never overwrite each other). A record holds:

- **The run.** `runId` (the search session id), `conversationId`, the user's `question`, `startedAt`/`finishedAt`,
  the planner's `searchQueries`, selected `datasources`, `web`, and the `reranker` provider/model when one ran.
- **Every candidate.** Each distinct chunk (identity: source + stored chunk text, the same key the citation checks
  use) any channel returned, cited or not. Chunk text is **not** stored: only `source`, a `chunkId` hash, the text
  length, and text signals computed at log time.
- **Observations.** One per (channel, query) hit: `stage` (`prefetch` or `tool`), `surface` (`minsync`,
  `datasource:<id>`, `web`, or the tool name), backend `method`, the `query`, the 1-based `rank` within that list,
  the channel's own raw `score`, and, for tool output that went through the merger or reranker, `rerankScore`,
  `retrievalMethodCount`, `sourceChunkCount`, `duplicateHitCount`. Prefetch lists are logged before interleaving,
  dedup, rerank, or cut.
- **Rerank.** The baseline reranker's rank and score for the candidates it kept.
- **Exposures.** Every place the model was shown the chunk: `baseline` (the first-answer prompt) or a tool result,
  with its 1-based position.
- **Text signals.** BM25 of the chunk against the question (over the run's pool) and the share of question terms the
  chunk contains.
- **Memory priors.** `priorSupport`: judged evidence from this source stored **before the run started** (the run's
  own judgments never leak into its features). `similarQuestionSupport`: judged evidence from this source among the
  similar past questions this run's memory context found; `null` when the context was never loaded.
- **Labels.** `citedFinal` (the final answer cites it after the relevance filter), `modelCited` (any answer of the run
  cited it before the filter), `relevance` (Jev's P(relevant) from the citation check), `support` (Jev's P(supports)
  from the memory gate). `judgments` says per run whether each Jev step `judged`, was `unavailable` (no label is
  guessed), or did `not-run`.

Runs on the `config`, `organize`, or uncited `extra` branches are not retrieval and are not logged. A failed write is
reported as a `ranking-log-failed` diagnostic; the answer is unaffected.

The log is on by default and stays on the machine. Turn it off with `"rankingLog": false` in `config.json` (or
`rankingLog: false` on `AutoRAGAgent`).

## Export

```bash
autorag memory export-ltr --output ./ltr [--valid-fraction 0.2]
```

Reads the log (never creates it) and writes LightGBM text files:

| File | Contents |
|------|----------|
| `train.tsv`, `valid.tsv` | Header row, `label` first, then the features in `manifest.featureNames` order; `nan` = missing |
| `*.tsv.query` | Rows per query group, in row order; LightGBM loads it automatically beside the data file |
| `*.tsv.position` | Best position the model was shown each row at (0 = never shown), for LightGBM's position-bias correction |
| `*.rows.jsonl` | `runId`, `chunkId`, `source`, `origin`, `label`, `position` per row, for error analysis |
| `manifest.json` | Schema version, feature names, `categoricalFeatures` (`origin_id`, `method_id`), train-fit `vocabularies`, label meanings, group/row/label counts, skipped groups, suggested LightGBM params |

Design decisions:

- **Graded labels 0–3.** 3 = cited by the final answer and P(supports) >= 0.7 (what memory stores); 2 = cited by the
  final answer; 1 = cited only by a draft and not found unrelated; 0 = everything else, including a citation Jev found
  unrelated (P(relevant) < 0.3). LightGBM's default `label_gain` (2^label − 1) fits.
- **Exposure is not a feature.** A chunk the model never saw cannot be cited, so "was it shown" would let the model
  learn the old ranker back. It goes to `.position` instead, which LightGBM's unbiased LambdaRank
  (`lambdarank_position_bias_regularization`) uses to separate position bias from relevance.
- **Hybrid channel representation (per-datasource calibration without schema churn).** External datasources never share
  chunks with each other and report incompatible score scales (BM25 magnitudes, cosine similarities, rank scores).
  Instead of either collapsing every datasource into one lossy bucket or adding wide columns per datasource, each row
  carries `origin_id` and `method_id` (LightGBM `categorical_feature`, vocabularies fit on `train` only in
  `manifest.vocabularies`) paired with `primary_rank`, `primary_raw_score`, and within-method `primary_norm_score`
  (0..1). Depth-2 tree splits learn datasource-specific score thresholds, while a newly connected datasource arrives as
  `nan` in `origin_id` and falls back cleanly to `origin_datasource`, `primary_norm_score`, and causal point-in-time
  `origin_prior_win_rate` / `origin_prior_candidates` (computed strictly over earlier finished runs) without breaking
  the column schema. Local files retain dedicated `dense_*`, `sparse_*`, and `hybrid_*` columns because dense vectors
  and SPLADE index the same chunks and can intersect on one candidate.
- **Missing is not zero.** A channel that did not find a chunk leaves its best rank and max score missing: rank 0
  would read as better than first. Hit counts and reciprocal-rank sums are genuinely 0.
- **Groups that teach nothing are dropped.** A run with one candidate, or with every candidate on the same label
  (e.g. the answer cited nothing), gives LambdaRank no gradient; the manifest counts them.
- **Split by whole run, in time order.** The newest runs are validation, so no query has rows on both sides and the
  validation set looks like the traffic a model would serve next.
- **Versioned.** `schemaVersion` is on every record; the store skips (and counts) a record of another version instead
  of mixing column meanings.

Train with LightGBM directly, for example:

```python
import json, lightgbm as lgb
manifest = json.load(open("ltr/manifest.json"))
cats = manifest["categoricalFeatures"]["indices"]
train = lgb.Dataset("ltr/train.tsv", params={"header": True, "label_column": 0}, categorical_feature=cats)
valid = lgb.Dataset("ltr/valid.tsv", params={"header": True, "label_column": 0}, reference=train, categorical_feature=cats)
model = lgb.train({"objective": "lambdarank", "metric": "ndcg", "eval_at": [1, 3, 5, 10]}, train, valid_sets=[valid])
```

## Known limits

- Labels come from one model's citations and one judge; they inherit both models' biases. `judgments` records when Jev
  did not judge so those runs can be filtered.
- Local files the model opened with `bash` and Jikji path hints are not candidates: they carry no channel score.
- Per-channel features are only as rich as each backend reports. MinSync `hybrid` (used by `search_all_documents`)
  reports one fused score; dense and SPLADE are separate only where they are queried separately.
- Nothing here trains or serves a model. Serving one would replace the merger/reranker order and needs its own
  offline evaluation against these exports first.
