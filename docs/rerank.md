# Reranking merged evidence

AutoRAG can reorder its merged retrieval evidence with a dedicated rerank model.
Reranking is a post-retrieval stage: the pipeline fans out across every
registered method, filters by trusted datasource access, merges and dedupes the
chunks (`ResultMerger`), and then the configured reranker reorders the survivors
by relevance to the query.

Reranking is **opt-in**. It runs only when you name a rerank model in the
`rerank` block (`rerank.model`); with no block, `"rerank": false`, or a block
without a model, AutoRAG does not call a reranker and sends none of your text to
one. Two providers are supported: the hosted **Dazzi Models** plan (`dazzi`,
using your `/login` credential) and **OpenRouter** (`openrouter`, a plain HTTPS
call to its rerank endpoint with your own key). `model` is the provider's own
rerank model id; AutoRAG ships no default, so pick one from your provider's
model list.

Local-first is no longer a contract: the local MinSync embedder remains the
zero-configuration default, and remote reranking (or a remote
`minSync.embedder`) is an explicit, trusted-config opt-in. Nothing falls back between providers silently.

## Configuration

```json
{
  "rerank": {
    "provider": "openrouter",
    "model": "<rerank model id from your provider>",
    "apiKeyEnv": "OPENROUTER_API_KEY",
    "topN": 25
  }
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | boolean | `true` when the block names a `model` | `false` disables reranking |
| `provider` | string | `openrouter` | Rerank provider id — `openrouter` or `dazzi` (any other value is a config error) |
| `model` | string | none (required) | Provider wire model id, for example the model your provider lists for reranking. Reranking runs only when this is set |
| `apiKeyEnv` | string | `OPENROUTER_API_KEY` (`openrouter`); not supported for `dazzi` | Env var holding the provider API key (never the secret itself). `dazzi` uses the credential stored by `/login`, and setting `apiKeyEnv` with it is a config error |
| `baseUrl` | string | provider default | Override the provider base URL (e.g. a proxy or self-hosted endpoint); for `dazzi` this is the API base **including** `/v1` |
| `topN` | positive integer | `25` | Keep only the top N merged results after reranking |
| `timeoutMs` | positive integer | `60000` | Per-request timeout. A reranker that does not answer within it fails the call, which is reported and falls back to retrieval order. The hosted reranker takes about 15 s for 172 documents and 45 s for 300 |

Setting `"rerank": false` disables reranking. Reranking also stays off when the
block names **no `model`**: there is no default model, so a block that only sets a
provider (what older versions of `autorag init` wrote) does nothing, and the
response carries one `rerank-disabled` warning that says to set `rerank.model`.
`autorag init` writes no `rerank` block at all.

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
- The **pre-fast-answer baseline** depends on whether reranking is on.
  - **Reranking on** (a `rerank.model` is named): the prefetch pool (Jikji answer
    paths, MinSync chunks, and chunks from datasources the Jev datasource check
    selected; by default up to 30 / 100 / 20 per source) is reranked down to
    `topN` (default 25) and injected as a single relevance-ordered section. If the
    reranker is unavailable or fails, the failure is reported (a `rerank-failed`
    warning on the response and a visible notice in the TUI, never shown to the
    model) and the baseline falls back to **retrieval order** cut to `topN`,
    taking ranks round-robin across the sources.
  - **Reranking off** (the default): nothing is reordered and the pool is not
    widened. Each source (Jikji, MinSync, and every selected datasource) is asked
    for its **own top 10** and shows those 10 in its own section, in retrieval
    order. A source with fewer than 10 hits shows what it has. Raise or lower the
    per-source count with `limits.prefetch.jikjiTopK`, `limits.prefetch.minSyncTopK`,
    and `limits.prefetch.datasourceTopK`. The model's own search tools
    (`search_all_documents`, `search_datasource_*`) keep their usual defaults.

  When the Jev query pipeline decomposes a local question (see
  [Jev Decisions](jev-decisions.md)), every sub-query asks each source again, the
  hits are interleaved and deduplicated per source, and (with reranking on) the
  whole pool is reranked against the **original** question.
- Single-datasource searches (`search_datasource_*`) are **not** model-reranked
  on their own: they already target one connection, so their merged order is
  kept as-is (their pre-fast-answer chunks are reranked as part of the pool).
- Each reranked result keeps its original `source` and `id`; its `score`
  becomes the provider's relevance score, and `metadata` gains
  `rerankProvider`, `rerankModel`, and `rerankScore`.
- Results whose `content` is empty or whitespace-only are **never sent** to the
  provider (the hosted reranker rejects an empty document with HTTP 400 and
  fails the whole request). They are not scored: with `topN` unset they follow
  the ranked results in their original order; with `topN` set they are dropped.
  If every result is blank no request is made.
- Documents longer than **131,000 characters** are cut to that length in the request
  only (the hosted reranker rejects 131,072 or more with HTTP 400); the returned
  results keep their full `content`. The upstream also enforces a token limit,
  so very long repetitive or CJK text can still be rejected below that length;
  such a failure is reported as `rerank-failed`.
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
