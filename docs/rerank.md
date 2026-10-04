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
    "topN": 20
  }
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | boolean | `true` when the block is present | `false` disables reranking |
| `provider` | string | `openrouter` | Rerank provider id |
| `model` | string | `voyageai/rerank-3-lite` | OpenRouter wire model id |
| `apiKeyEnv` | string | `OPENROUTER_API_KEY` | Env var holding the provider API key (never the secret itself) |
| `baseUrl` | string | OpenRouter default | Override the provider base URL (e.g. a gateway or self-hosted endpoint) |
| `topN` | positive integer | unset (all results) | Return only the top N merged results |
| `timeoutMs` | positive integer | SDK default | Per-request timeout |

Setting `"rerank": false` disables reranking. A config file with no `rerank`
block leaves the stage off; `autorag init` writes the block above so new
configurations have it.

The API key is read from the environment at call time. Set it before starting
AutoRAG:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

## Behavior

- The reranker runs **after** merge/dedup, so it reorders the complete distinct
  evidence pool. With no `topN` it returns every distinct chunk, reordered; with
  `topN` it truncates to the most relevant N.
- Each reranked result keeps its original `source` and `id`; its `score`
  becomes the provider's relevance score, and `metadata` gains
  `rerankProvider`, `rerankModel`, and `rerankScore`.
- A configured-but-unavailable reranker (missing key) or a rerank failure is
  reported as a `rerank-failed` diagnostic and the merged order is preserved.
  A reranker outage never hides retrieval evidence.

## Local reranking

`baseUrl` can point at any Cohere-compatible `/v1/rerank` endpoint, including a
local server. The `Reranker` interface (`src/retrieval/rerank.ts`) is
provider-agnostic: a local reranker implements `describe()` + `rerank()` and
plugs into `RetrievalEngine` / `AutoRAGAgent` without touching the pipeline.
