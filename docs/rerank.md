# Reranking merged evidence

AutoRAG can reorder its merged retrieval evidence with a dedicated rerank model.
Reranking is a post-retrieval stage: the pipeline fans out across every
registered method, filters by trusted datasource access, merges and dedupes the
chunks (`ResultMerger`), and then the configured reranker reorders the survivors
by relevance to the query.

The default reranker routes through **OpenRouter** using the
[`@openrouter/sdk`](https://www.npmjs.com/package/@openrouter/sdk) TypeScript
SDK and the [`voyageai/rerank-3-lite`](https://openrouter.ai/voyageai/rerank-3-lite)
model.

Local-first is no longer a contract: the local embedding gateway remains the
zero-configuration default, and remote reranking (or remote embedding) is an
explicit, trusted-config opt-in. Nothing falls back between providers silently.

## Configuration

```json
{
  "rerank": {
    "provider": "openrouter",
    "model": "voyageai/rerank-3-lite",
    "apiKeyEnv": "OPENROUTER_API_KEY",
    "topN": 25
  }
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | boolean | `true` when the block is present | `false` disables reranking |
| `provider` | string | `openrouter` | Rerank provider id (`openrouter` is the only supported id; any other value is a config error) |
| `model` | string | `voyageai/rerank-3-lite` | OpenRouter wire model id |
| `apiKeyEnv` | string | `OPENROUTER_API_KEY` | Env var holding the provider API key (never the secret itself) |
| `baseUrl` | string | OpenRouter default | Override the provider base URL (e.g. a gateway or self-hosted endpoint) |
| `topN` | positive integer | `25` | Keep only the top N merged results after reranking |
| `timeoutMs` | positive integer | SDK default | Per-request timeout |

Setting `"rerank": false` disables reranking. A config file with no `rerank`
block leaves the stage off; `autorag init` writes the block above so new
configurations have it.

The API key is read from the environment when the reranker is created. Set it
before starting AutoRAG:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

## Behavior

- The reranker runs **after** merge/dedup, so it reorders the distinct evidence
  pool and keeps the top `topN` (default **25**). `search_all_documents` merges
  up to 500 chunks; `AutoRAGLite.retrieve` uses the same ceiling.
- The **pre-fast-answer baseline** is reranked too: the
  prefetch pool (Jikji answer paths ≤100 + MinSync chunks ≤100 + chunks from
  datasources the Jev datasource check selected ≤100) is reranked down to
  `topN` (default 25) and injected as a single relevance-ordered section, so
  the immediate answer is grounded in relevance order rather than raw method
  order. If the reranker is unavailable or fails, the unranked
  Jikji/MinSync/datasource sections are used and the fast answer is
  unaffected. When the Jev query pipeline decomposes a local question (see
  [Jev Decisions](jev-decisions.md)), every sub-query's Jikji, MinSync, and
  selected-datasource hits are interleaved and deduplicated into one pool with
  the same per-source caps, and that pool is reranked against the **original**
  question.
- Single-datasource searches (`search_datasource_*`) are **not** model-reranked
  on their own: they already target one connection, so their merged order is
  kept as-is (their pre-fast-answer chunks are reranked as part of the pool).
- Each reranked result keeps its original `source` and `id`; its `score`
  becomes the provider's relevance score, and `metadata` gains
  `rerankProvider`, `rerankModel`, and `rerankScore`.
- A configured-but-unavailable reranker (missing key) or a rerank failure is
  reported as a `rerank-failed` diagnostic and the merged order is preserved.
  A reranker outage never hides retrieval evidence.

## Local reranking

`baseUrl` can point at any Cohere-compatible `/v1/rerank` endpoint, including a
local server. `openrouter` is the only supported `provider` id; any other value
is rejected as a configuration error. For an in-process implementation, pass a
custom `Reranker` to `RetrievalEngine` through its `reranker` option.
