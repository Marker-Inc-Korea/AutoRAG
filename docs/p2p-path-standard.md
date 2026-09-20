# P2P source identifier standard

One canonical source identifier flows through retrieval, policy, and the P2P
wire. This document is the normative reference for that standard across the
local, datasource, and policy layers.

## Canonical forms

| Layer | Form | Example |
|-------|------|---------|
| Local file (virtual id) | `/<root-prefix>/<relative-path>` | `/docs/policy/refund.md` |
| Local file (retrieval `source`) | OS-absolute real path | `/home/me/docs/policy/refund.md` |
| Datasource identity | `/<kind>/<instance>/...` slash hierarchy | `/kakao/personal/chunks/chunk-1` |

Retrieval methods return local files as OS-absolute real paths (readable,
`fs.existsSync`-able) and datasource results as slash-hierarchical identities
that are **not** filesystem paths. The retired `kakao:<chat>/<sender>/<chunk>`
colon scheme is not emitted anywhere and is rejected by normalization.

## `normalizeSource` — the single normalization point

`normalizeSource(source, sourceRoots)` in `src/filesystem/source-paths.ts`
converts any of the three forms to the canonical virtual id:

1. An id already inside a configured root's virtual namespace passes through
   validation unchanged.
2. An OS-absolute real path contained in a configured source root becomes that
   root's virtual id (`/<prefix>/<relative>`); the longest containing root
   wins for nested roots.
3. A datasource slash identity (a slash-prefixed id whose first segment is not
   a well-known filesystem root) passes through validation unchanged.
4. An absolute filesystem path outside every configured root passes through
   as its own canonical form — backslashes are converted to forward slashes
   so Windows drive/UNC paths canonicalize identically on every host —
   letting operators write policy globs against real absolute paths (for
   example `/etc/secrets/**` or `C:/secrets/**`).

Everything else fails closed as `undefined`: traversal segments (`..`), URL
schemes (including the retired colon scheme and `file:`), backslashes in
non-absolute sources, and empty input.

A local file name is data, not URL syntax: `#` and `?` are ordinary
characters there and stay in the canonical id. Only a *scheme* makes a
source URL-shaped. The rejection targets syntax, never a character that a
real file system permits.

## Policy resolution normalizes before matching

`PolicyStore` accepts `sourceRoots` (built with `planSourceRoots(searchPaths)`).
When configured, `resolvePolicy`, `markSourceSeen`, `promoteSource`, and
`isSourceSeen` canonicalize every incoming source through `normalizeSource`
before glob matching or seen-source lookup:

- An absolute real path under `searchPaths` matches the root's virtual globs
  (`/docs/**`).
- A datasource identity matches its namespace globs (`/kakao/**`).
- An absolute real path outside `searchPaths` matches absolute-path globs as
  itself.
- A syntactically invalid source (traversal, scheme, backslash) resolves to
  `private` (fail-closed).

Without `sourceRoots`, `PolicyStore` matches sources exactly as provided
(unchanged legacy behavior). `autorag serve` always configures `sourceRoots`
and passes the store's resolver directly to the peer server.

## `autorag serve` startup readiness check

`autorag serve` prints a non-fatal stderr warning when MinSync retrieval is
not ready (`configured` / `degraded` / `unavailable` — anything but `ready`)
before accepting peer queries. Peer queries answered from an unready index
look like empty corpora; refresh first with
`autorag refresh --method minsync`.

## Empty-result fallback for remote sessions

A remote (peer-serving) session that completes without the agent emitting
structured results no longer throws. Instead `searchDocuments` resolves a
structured empty response:

- `results: []`, answer `"No verified results were found for this query."`
- diagnostic `{ code: "no-verified-results", severity: "info" }`

The egress gate maps that diagnostic to a wire response with a
`no-verified-results` diagnostic code, so the requesting peer can distinguish
"nothing found" from `policy-denied`, `internal-error`, and timeouts. Local
sessions keep throwing `completed without emitting structured results`.
