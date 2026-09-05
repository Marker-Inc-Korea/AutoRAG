# Competitive Landscape Study — September 2026

> Observation date: **2026-09-05**. Method: multi-lane OSS landscape scan (local
> CLI+MCP engines, agent/RAG platforms, backend MCP servers, connector
> engines) with first-party source verification (repos, docs, licenses,
> release dates) and an adversarial red-team round against every gap claim.
> This document records the findings behind the three
> [core values](../README.md#core-values) — especially value 1, "never
> migrate your data to search it."

## Scope of the question

The study asked: **does any public OSS already ship the proposed AutoRAG
bundle** — harness-free federation of independent third-party CLI-owned
stores, with source-native identities/ACL, exposed to both MCP clients and a
standalone search agent?

## Verdict

- **Under a strict definition: no exact incumbent was found.** An 18-angle
  counter-search plus a red-team round identified no public project proving
  all five parts (federation, third-party CLI-owned stores, source-native
  identity/ACL, model-safe tool surface, native MCP).
- **Every broad version of the claim is refuted.** Do not market "no
  incumbent" without the strict definition.

## What is commoditized (do not build the product around these)

| Claimed differentiator | Refuted by |
|---|---|
| MCP-native access | QMD v2.8.3 (MIT, CLI+MCP), Quarry v3.2.1 (MIT, daemon+MCP), msgvault (first-party MCP), official MCP servers from Elastic (Agent Builder, 9.3 GA), OpenSearch (`/_plugins/_ml/mcp` 3.3+), Qdrant, Weaviate (`/v1/mcp`), Milvus/Zilliz, Meilisearch |
| Local-first operation | QMD, Quarry, mcp-local-rag (383★), Kotaemon, msgvault (fully offline) |
| Hybrid retrieval (BM25 + vector + RRF) | QMD (SQLite FTS5 + local GGUF + RRF), msgvault (FTS5 + Ollama + RRF), Quarry (Tantivy + LanceDB) — a published, copyable recipe |
| CLI surface | AnythingLLM (`any`), MFS (`mfs`), msgvault, QMD |
| "No agent harness exists" | Negative space; absence of a feature is not user value |

## Strongest competitors (as of 2026-09-05)

- **Zilliz MFS** ([zilliztech/mfs](https://github.com/zilliztech/mfs),
  Apache-2.0) — the most dangerous single competitor: named vendor
  (the Milvus company), production-grade index pipeline, broad connector
  catalog. **Structural difference:** it is server-first — a thin `mfs` CLI
  over a stateful `mfs-server` that ingests every source into its own
  index/storage. MFS cannot adopt AutoRAG's value 1 without abandoning its
  own architecture.
- **msgvault** ([kenn-io/msgvault](https://github.com/kenn-io/msgvault), MIT)
  — near-exact within the messages/meetings domain: local-first, offline,
  FTS5 + optional local-embedder hybrid search, first-party MCP, agent
  skills, source-scoped identity/person model. Treated as a **resource, not
  a rival** — integration tracked in
  [#1530](https://github.com/Marker-Inc-Korea/AutoRAG/issues/1530).
- **QMD v2.8.3 / Quarry v3.2.1** (both MIT) — strong local CLI+MCP engines,
  but each owns a single generic local index rather than orchestrating
  third-party CLI stores.
- Platform incumbents (AnythingLLM v1.16.1, RAGFlow v0.27.1, Onyx, Open
  WebUI + oikb, Khoj, pyLLMSearch) bundle a harness/UI/answer generation and
  are not agent-neutral datasource control planes.

## Why "never migrate your data to search it" survives

Every competitor either (a) forces sources into its own server/index (MFS),
(b) covers one domain with one self-owned store (msgvault, QMD), or (c)
wraps retrieval in its own harness (platform incumbents). The surviving
differentiator is contract-level, not feature-level:

1. CLI-owned stores stay where they are — no central index copy, no forced
   upload;
2. results carry human-readable, source-native identities with
   scope-checked access (default deny; the model can narrow but never widen
   scopes);
3. secrets never leave the tool that owns them;
4. per-CLI failure isolation (a missing binary degrades to diagnostics).

Replicating this retroactively requires re-architecting a competitor's store
schema — MCP, local-first, or hybrid retrieval can be copied in weeks; this
interface contract cannot.

## Honest risk register

- msgvault or MFS could close the strict-definition gap in a single release
  (MFS first-party MCP in core remains unverified at observation time).
- Federation wrappers around existing crawlers/CLIs may surface after this
  scan.
- "No exact incumbent" is a point-in-time negative tied to the observation
  date and the five-part definition above.

## What this means for contributors

- Reject or reshape PRs that force data migration into an AutoRAG-managed
  store (see [AGENTS.md](../AGENTS.md#product-positioning), value 1).
- Do not lead with MCP support, local-first, or hybrid retrieval in
  messaging — every serious competitor has all three.
- Lead with provenance, source-native identity/ACL, and harness-free
  federation; pair them with "just works" installation and local, fast
  retrieval.
