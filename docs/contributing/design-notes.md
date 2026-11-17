---
title: "Design notes"
description: "Rationale behind two decisions: why AutoRAG federates CLI-owned stores in place, and what a redesigned peer-sharing feature must keep."
---

# Design notes

This page is **rationale, not behavior**. It records why two design directions were chosen, so a future change does not silently undo a load-bearing decision. When a note here and the code disagree, the code is authoritative; treat the note as the intent behind it.

## Why in-place federation

**Decision.** AutoRAG federates CLI-owned stores (`lazykatok`, `discrawl`, the crawler-backed skills, `mailcrawl`) in place. It does not ingest sources into a central index it owns, and it does not hold a copy of the corpus.

This is core value 1: never migrate your data to search it. Results carry source-native identities such as `/kakao/<instance>/chunks/<chunk>`, secrets stay with the tool that owns them, and one failing CLI degrades to diagnostics rather than crashing the search loop.

**What is commoditized (do not build the product around it).** MCP-native access, local-first operation, and hybrid retrieval (sparse + vector fusion) are common; a serious competitor has all three and they can be copied in weeks. Do not lead the project's messaging with any of them.

**What is durable.** The surviving differentiator is contract-level, not feature-level:

1. CLI-owned stores stay where they are — no central index copy, no forced upload.
2. Results carry human-readable, source-native identities, and an ordinary query scope can narrow a scope-capable datasource's results but never widen visibility.
3. Secrets never leave the tool that owns them.
4. Per-CLI failure isolation: a missing binary degrades to diagnostics.

Replicating this retroactively means re-architecting a competitor's store schema, so it is far harder to copy than MCP or a retrieval recipe.

**Consequences for contributors.** Reject or reshape pull requests that force data migration into an AutoRAG-managed store. Keep result sources opaque and slash-hierarchical, never fake OS-absolute paths (see [Adding a datasource](new-datasource.md)). Keep failure isolation per CLI, and report a CLI's own error verbatim instead of returning an empty set.

## Peer-sharing redesign lessons

AutoRAG previously shipped a SimpleX-based peer-sharing surface (the removed `serve` and peer-to-peer commands, the `src/p2p/` tree, and the `remoteSession` agent mode, including the `query_peer_agent` and `recommend_peer_targets` tools). It did not work reliably and was removed so a redesign can start clean. This note records what a redesign must keep; it is not a spec.

### Threat model

A peer is an authenticated-but-untrusted party that can send a natural-language query to a librarian and read the answer. Three things must never leave the machine unless the owner allowed them: document bytes, text from private datasources, and anything a prompt-injected model was tricked into emitting.

### Decisions a redesign must keep

- **Default private, per-source policy.** Every source resolves to a tier (`private`, `never`, `always`, `peers`) with a glob table. An unmatched or unparseable source resolves to `private`. New files stay private until explicitly promoted.
- **Deny at retrieval and again at egress.** The policy filters retrieval results, and the response builder re-checks every returned source against the policy for that peer. A source the model cited that retrieval never returned is dropped (`source-unmappable`). Two layers, because the model is the untrusted component between them.
- **One canonical source id.** Retrieval returns OS-absolute paths for local files and `/<kind>/<instance>/...` slash identities for datasources; policy globs matched the canonical virtual id (`/docs/**`), never a raw path. Anything with traversal, a URL scheme, a backslash, or empty input failed closed. `normalizeSource`, `normalizeVirtualPath`, and `isFilesystemAbsolutePath` remain in `src/filesystem/source-paths.ts`.
- **Remote sessions are a different agent, not a flag on the local one.** The old `remoteSession` flag stripped `edit`/`write`, web search, Everything, FSearch, Jev, self-config, retrieval memory, and local-file citation — a list long enough that it belongs in a separate tool allowlist, not scattered branches through `agent.ts`.
- **Injection checks on both sides.** Inbound: length cap, NFC and confusable folding, stripped format characters, encoded-blob detection, instruction signatures, and an optional model classifier. Outbound: the same signatures plus a workspace-path leak scan over the answer and every excerpt.
- **PII redaction is best-effort hygiene.** Deterministic email/phone/RRN/card patterns plus an optional NER pass behind a caller-supplied model function. Never documented as a compliance guarantee.
- **Human approval before egress.** Responses could wait in a pending queue (`requests approve|deny`). Denial returned a refusal with no document content.
- **Quotas and bounded concurrency.** A token bucket per contact (10 queries/hour, burst 3), a single-flight serializer with a bounded queue, and 64 KiB body / 25 MiB file caps. Refusals were a closed set: `auth-error`, `replay-rejected`, `rate-limited`, `injection-detected`, `policy-denied`, `queue-full`, `internal-error`.
- **Empty is a result.** A remote run that emitted nothing returned a structured `no-verified-results`, never `internal-error`. Serve warned at startup when MinSync was not `ready`, because peers cannot tell an empty corpus from an unready index.

### What to solve differently

- **Identity.** Trust was keyed on a SimpleX contact id plus a local alias, and the transport was the least reliable part. Pick a transport whose failure modes are observable, and keep identity separate from it.
- **Coupling.** Policy options rode on `RetrievalOptions` through every retrieval path. A server-side wrapper that owns the run and its observed-source registry would keep the local agent free of them.
- **Asynchronous answers.** Returning "pending" after a timeout and resuming through a persisted callback was hard to reason about and to test. Decide up front whether peer queries are synchronous or an inbox.
- **Testing.** Live SimpleX QA needed two real profiles on loopback. A fake transport behind a narrow interface carried most of the value; keep the boundary small enough to fake.

### State left behind

A `p2p` key in `config.json` is ignored, not rejected, so existing configs keep loading — nothing reads it. Old state under `<workspace>/.autorag/p2p/` and `$AUTORAG_HOME` (policy, peer registry, approvals, SimpleX databases) is unused and can be deleted by the owner; AutoRAG does not touch it.

## Related pages

- [Architecture](architecture.md) for the retrieval pipeline these decisions run inside.
- [Adding a datasource](new-datasource.md) for the in-place federation contract.
- [Development workflow](development.md) for the guardrails around datasource and P2P changes.
