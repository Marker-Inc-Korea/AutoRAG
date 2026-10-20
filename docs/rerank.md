# Reranking merged evidence

AutoRAG can reorder its merged retrieval evidence with a dedicated rerank model.
Reranking is a post-retrieval stage: the pipeline fans out across every
registered method, filters by trusted datasource access, merges and dedupes the
chunks (`ResultMerger`), and then the configured reranker reorders the survivors
by relevance to the query.

The default reranker is the hosted **Dazzi Models** reranker
(`qwen/qwen3-reranker-8b`) when you are signed in (`/login`), and
**OpenRouter** (a plain HTTPS call to its rerank endpoint with
[`voyageai/rerank-3-lite`](https://openrouter.ai/voyageai/rerank-3-lite))
otherwise. The choice is made by `autorag init` when it writes the `rerank`
block; a config file with no `rerank` block leaves reranking off, and an
existing block keeps what it says.

Local-first is no longer a contract: the local MinSync embedder remains the
zero-configuration default, and remote reranking (or a remote
`minSync.embedder`) is an explicit, trusted-config opt-in. Nothing falls back between providers silently.

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
| `provider` | string | `openrouter` | Rerank provider id — `openrouter` or `dazzi` (any other value is a config error) |
| `model` | string | `voyageai/rerank-3-lite` (`openrouter`) / `qwen/qwen3-reranker-8b` (`dazzi`) | Provider wire model id |
| `apiKeyEnv` | string | `OPENROUTER_API_KEY` (`openrouter`); not supported for `dazzi` | Env var holding the provider API key (never the secret itself). `dazzi` uses the credential stored by `/login`, and setting `apiKeyEnv` with it is a config error |
| `baseUrl` | string | provider default | Override the provider base URL (e.g. a proxy or self-hosted endpoint); for `dazzi` this is the API base **including** `/v1` |
| `topN` | positive integer | `25` | Keep only the top N merged results after reranking |
| `timeoutMs` | positive integer | `60000` | Per-request timeout. A reranker that does not answer within it fails the call, which is reported and falls back to retrieval order. The hosted reranker takes about 15 s for 172 documents and 45 s for 300 |

Setting `"rerank": false` disables reranking. A config file with no `rerank`
block leaves the stage off; `autorag init` writes the block above so new
configurations have it.

The API key is read from the environment when the reranker is created. Set it
before starting AutoRAG:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

Alternatively, `provider: "dazzi"` reranks through the hosted Dazzi Models
plan using the credential stored by `/login` and the server root at
`AUTORAG_BASE_URL` (default `https://api.dazziapp.com`). There is no API-key
environment variable for it; sign in with `autorag tui` and `/login` first:

```json
{
  "rerank": {
    "provider": "dazzi",
    "model": "qwen/qwen3-reranker-8b"
  }
}
```

The rerank client appends `/rerank` to the base URL, so an explicit
`rerank.baseUrl` for `dazzi` is the API base **including** `/v1`
(`https://api.dazziapp.com/v1`); the base derived from `AUTORAG_BASE_URL` gets
`/v1` appended unless it already ends in `/v1`.

## Behavior

- The reranker runs **after** merge/dedup, so it reorders the distinct evidence
  pool and keeps the top `topN` (default **25**). `search_all_documents` merges
  up to 500 chunks; `AutoRAGLite.retrieve` uses the same ceiling.
- The **pre-fast-answer baseline** is reranked too: the
  prefetch pool (Jikji answer paths ≤100 + MinSync chunks ≤100 + chunks from
  datasources the Jev datasource check selected ≤100) is reranked down to
  `topN` (default 25) and injected as a single relevance-ordered section, so
  the immediate answer is grounded in relevance order rather than raw method
  order. If the reranker is unavailable or fails, the failure is reported (a
  `rerank-failed` warning on the response and a visible notice in the TUI, never
  shown to the model) and the baseline falls back to **retrieval order** cut to
  `topN`, taking ranks round-robin across the Jikji, MinSync, and datasource
  lists. With reranking disabled the sections stay uncut. When the Jev query
  pipeline decomposes a local question (see
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

## Wire format

The client is a plain `fetch` POST to `<base>/rerank` with
`{ model, query, documents, top_n }` and a bearer token. It reads only
`results[].index` and `results[].relevance_score` from the response, so any
endpoint with that shape works, including a self-hosted or gateway server set
through `baseUrl`. `openrouter` and `dazzi` are the only supported `provider`
ids; any other value is rejected as a configuration error. For an in-process
implementation, pass a custom `Reranker` to `RetrievalEngine` through its
`reranker` option.
