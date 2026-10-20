# P2P sharing: what was removed and what a redesign must keep

AutoRAG used to ship SimpleX-based peer sharing (`autorag serve`, `autorag p2p`,
the `query_peer_agent` / `recommend_peer_targets` tools, `src/p2p/`, and the
`remoteSession` agent mode). It did not work reliably and was removed so a
redesign can start clean. The last commit that contains the implementation is
`fba1ba95` (`git show fba1ba95:src/p2p/<file>`).

This note records the design decisions worth carrying forward. It is not a spec.

## Threat model the old design assumed

A peer is an authenticated-but-untrusted party that can send a natural-language
query to your librarian and read the answer. Three things must never leave the
machine unless the owner allowed them: document bytes, text from private
datasources, and anything a prompt-injected model was tricked into emitting.

## Decisions to keep

- **Default private, per-source policy.** Every source resolved to a tier
  (`private`, `never`, `always`, `peers`) with a glob table. An unmatched or
  unparseable source resolved to `private`. New files stayed private until
  explicitly promoted (`newFilesPublic` was the opt-in).
- **Deny at retrieval and again at egress.** The policy filtered retrieval
  results, and the response builder re-checked every returned source against the
  policy for that peer. A source the model cited that retrieval never returned
  was dropped (`source-unmappable`). Two layers, because the model is the
  untrusted component between them.
- **One canonical source id.** Retrieval returns OS-absolute paths for local
  files and `/<kind>/<instance>/...` slash identities for datasources.
  Policy globs matched the canonical virtual id (`/docs/**`), never a raw path.
  Anything with traversal, a URL scheme, a backslash, or empty input failed
  closed. `normalizeSource`, `normalizeVirtualPath`, and `isFilesystemAbsolutePath`
  remain in `src/filesystem/source-paths.ts`; file names keep `#` and `?` (issue #1617).
- **Remote sessions are a different agent, not a flag on the local one.** The
  old `remoteSession` flag removed `edit`/`write`, web search, Everything,
  FSearch, Jev, self-config, retrieval memory, and local-file citation. That
  list was long enough that it belonged in a separate tool allowlist, not
  scattered `if (remote)` branches through `agent.ts`.
- **Injection checks on both sides.** Inbound: length cap, NFC and confusable
  folding, stripped format characters, encoded-blob detection, instruction
  signatures, and an optional model classifier. Outbound: the same signatures
  plus a workspace-path leak scan over the answer and every excerpt.
- **PII redaction is best-effort hygiene.** Deterministic email/phone/RRN/card
  patterns, optional NER pass behind a caller-supplied model function. Never
  documented as a compliance guarantee.
- **Human approval before egress.** Responses could wait in a pending queue
  (`requests approve|deny`). Denial returned a refusal with no document content.
- **Quotas and bounded concurrency.** Token bucket per contact (10 queries/hour,
  burst 3), single-flight serializer with a bounded queue, 64 KiB body and
  25 MiB file caps. Refusals were a closed set: `auth-error`, `replay-rejected`,
  `rate-limited`, `injection-detected`, `policy-denied`, `queue-full`,
  `internal-error`.
- **Empty is a result.** A remote run that emitted nothing returned a structured
  `no-verified-results`, never `internal-error`. Peers cannot tell an empty
  corpus from an unready index, so serve warned at startup when MinSync was not
  `ready`.

## Problems the redesign should solve differently

- **Identity.** Trust was keyed on a SimpleX contact id plus a local alias, and
  the transport (a spawned `simplex-chat` WebSocket server) was the least
  reliable part. Pick a transport whose failure modes are observable, and keep
  identity separate from it.
- **Coupling.** Policy options (`peerFingerprint`, `resolvePolicy`,
  `observedSources`) rode on `RetrievalOptions` through every retrieval path.
  A server-side wrapper that owns the run and its observed-source registry
  would keep the local agent free of them.
- **Asynchronous answers.** `query_peer_agent` returned "pending" after 60 s and
  resumed through a persisted callback. That was hard to reason about and to
  test; decide up front whether peer queries are synchronous or an inbox.
- **Testing.** Live SimpleX QA needed two real profiles on loopback. A fake
  transport behind a narrow interface carried most of the value; keep the
  boundary small enough to fake.

## Config and state left behind

- A `p2p` key in `config.json` is now ignored, not rejected, so existing
  configs keep loading. Nothing reads it.
- Old state under `<workspace>/.autorag/p2p/` and `~/.autorag` (policy, peer
  registry, approvals, SimpleX databases) is unused and can be deleted by the
  owner. AutoRAG does not touch it.
