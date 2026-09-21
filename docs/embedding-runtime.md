# Shared embedding runtime

AutoRAG provides one local embedding service for MinSync and supported native
datasource integrations. The service owns embedding computation only. MinSync,
discrawl, and other datasource CLIs retain ownership of their archives, chunks,
vector stores, generations, and source identities.

The service binds to `127.0.0.1` and does not silently fall back to a remote
embedding provider. Model weights are mutable user-cache data and are never
included in the npm package or source tree.

## Profiles

The default profile is `qwen3-embedding-0.6b`:

| Profile | Provider | Model asset | Dimension | Query prefix | Passage prefix | License |
|---|---|---|---:|---|---|---|
| `qwen3-embedding-0.6b` (default) | Qwen | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 1024 | empty | empty | Apache-2.0 |
| `embeddinggemma-300m` | Google | `embeddinggemma-300M-Q8_0.gguf` | 768 | `task: search result \| query: ` | `title: none \| text: ` | Gemma Terms of Use |

Profile identity includes the provider, model, immutable model revision,
dimension, query prefix, passage prefix, and llama.cpp runtime build. A profile
change is an embedding identity change, not an in-place vector-store setting
change.

EmbeddingGemma is available as a selectable profile. Its terms and required
flow-down text are in [`licenses/`](../licenses/), and legal approval remains a
release gate.

## Cache, import, and offline operation

The default cache root is `~/.autorag` on macOS and `%USERPROFILE%\\.autorag` on
Windows. `AUTORAG_HOME` overrides the root. The runtime state uses this layout:

```text
<autorag-home>/
  models/
    Qwen3-Embedding-0.6B-Q8_0.gguf
    embeddinggemma-300M-Q8_0.gguf
    <verified platform runtime archive>
  runtime/                         # reserved runtime-resolution root
  embedding-runtime.lock
  embedding-runtime.pid
  embedding-runtime.log
```

The current cache downloader places downloaded assets under `models/`; the
`runtime/` root is the separate runtime-resolution namespace used by the
supervisor. Downloads are written to a `.part` file, verified with SHA-256, and
atomically renamed. A corrupt existing file is removed before a new attempt.
An import uses the same hash check and atomic placement; it does not copy an
unverified weight into the cache.

Offline mode disables network downloads. `ensureRuntime` can run offline when
the selected model and platform runtime are already present and verified;
otherwise it returns an `offline-missing` cache diagnostic. `models verify`
verifies a cached model without allowing a download.

The model commands are:

```bash
autorag models prefetch [--profile qwen3-embedding-0.6b]
autorag models import /path/to/model.gguf --profile embeddinggemma-300m
autorag models verify --profile embeddinggemma-300m
```

`import` accepts only a file whose SHA-256 matches the pinned profile asset.
The runtime asset is selected for the current supported platform and is fetched
from the pinned release URL at release time; model weights remain in the user
cache.

## Runtime and backend selection

The gateway supervises one `llama-server` child and one selected model. The
supported backend values are `auto`, `cpu`, and `vulkan`:

- `auto` uses the configured profile/platform default.
- `cpu` selects the CPU runtime for the selected platform.
- `vulkan` selects the Windows x64 Vulkan runtime and does not mean that a
  non-Windows platform has a Vulkan asset in this release.

The runtime API accepts the platform and backend options. The current default
selection does not make an unbounded network probe or silently switch to a
remote provider.

The supported pinned runtime assets are macOS arm64 Metal, Windows x64 CPU, and
Windows x64 Vulkan. Windows ARM is not shipped. The exact URLs, revisions,
hashes, archive members, and notice mappings are in the machine-readable
manifest below and in [`../licenses/embedding-assets.json`](../licenses/embedding-assets.json).

## Gateway protocol

The gateway chooses an ephemeral internal port, binds only to loopback, and
exposes these routes:

| Route | Request | Response |
|---|---|---|
| `GET /healthz` | no body | JSON health object with `status`, `backend`, `model`, `dimension`, `runtimeBuild`, and `profileId` |
| `POST /embed` | `{ "inputs": ["text", "text"] }` | ordered bare `number[][]` rows |
| `POST /v1/embeddings` | `{ "input": ["text", "text"] }` | OpenAI-compatible `{ "object": "list", "model": "...", "data": [{ "object": "embedding", "embedding": [...], "index": 0 }] }` |
| `POST /api/embeddings` | `{ "prompt": "text" }` | Ollama-compatible `{ "embedding": [...] }` |

The gateway validates non-empty string inputs, batch and body limits, response
row count, response ordering, vector dimension, and finite numeric values. It
rejects non-loopback upstream URLs before making a request. The upstream
`llama-server` endpoint is also loopback-only. The gateway returns structured
JSON errors with a `code`; common codes are `bad-request`, `batch-limit`,
`non-loopback-host`, `non-loopback-upstream`, `timeout`, `upstream`, and
`upstream-violation`.

The gateway is started on demand by the MinSync semantic path. It can be
inspected or stopped with:

```bash
autorag gateway status --format json
autorag gateway stop
```

`status --format json` reports the supervisor state, selected backend, model,
profile when known, process/port information when available, and a sanitized
health or failure object. Startup has a bounded readiness check and one
bounded restart. The supervisor records bounded stdout/stderr in
`embedding-runtime.log`, removes stale pid/lock state, and uses process-group
cleanup (or Windows `taskkill` fallback) during shutdown.

## MinSync default and migration

MinSync semantic sync and query use the AutoRAG gateway when no explicit
external endpoint is configured. The effective no-flag profile is
`qwen3-embedding-0.6b`, so a new workspace uses 1024-dimensional vectors
without Ollama, TEI, an API key, or an endpoint setting.

The MinSync workspace keeps its state in the `.minsync` subdirectory of
`<workspace>/.autorag/minsync` and records the runtime identity there, beside
the sync cursor and `config.toml`:

```text
<workspace>/.autorag/minsync/.minsync/autorag-embedding-identity.json
<workspace>/.autorag/minsync/.minsync/cursor.json
<workspace>/.autorag/minsync/.minsync/config.toml
```

A dimension or identity mismatch is detected before semantic vector reuse.
MinSync performs a full sync when the effective identity changes during sync;
query reports that a full reindex is required instead of mixing dimensions.
A failed rebuild does not replace the previous published generation.

### Existing 768-dimensional Ollama/TEI stores

An existing store created with Ollama or a TEI adapter at 768 dimensions must
not be reused with the default Qwen3 profile. Choose one of these explicit
paths:

1. Reindex the workspace for the new default profile:

   ```bash
   autorag index reset --method minsync --yes
   autorag refresh --method minsync --json
   ```

2. Pin the compatible 768-dimensional profile in trusted config and keep the
   workspace identity explicit:

   ```json
   {
     "minSync": {
       "embedder": {
         "profile": "embeddinggemma-300m"
       }
     }
   }
   ```

   Then run `autorag refresh --method minsync`. Pinning the profile preserves
   the 768-dimensional contract, but it does not make an Ollama/TEI vector
   store interchangeable with a store created by a different provider,
   prefix, model revision, or runtime identity. Reindex when the recorded
   identity does not match.

The migration diagnostic for the legacy path is:

```text
This workspace uses the legacy 768-dimensional Ollama/TEI embedding path. Reindex explicitly, or pin an explicit profile config before using the new default runtime.
```

If the gateway or model is unavailable, MinSync reports an
`embedder-unavailable` diagnostic and retains its lexical/BM25 lane. A hybrid
query can fall back to BM25; semantic-only callers receive the typed failure
rather than a false semantic result.

## Datasource compatibility

The shared runtime is a provider boundary, not a shared datasource store.

| Datasource | Shared-runtime compatibility | Ownership boundary |
|---|---|---|
| MinSync | Gateway default | MinSync owns `.minsync`, CDC chunks, vectors, and source mapping; AutoRAG supplies the gateway endpoint and identity. |
| discrawl | Managed native config | When `configPath` is not explicit and a workspace is available, AutoRAG writes `.autorag/datasources/discrawl/config.toml` with the marker `# AutoRAG managed discrawl embeddings v1` and native `[search.embeddings]` `provider`, `model`, `base_url`, and `dimensions`. discrawl owns SQLite, embeddings, FTS, and rebuilds. |
| lazykatok | Pending upstream provider contract | No shared-runtime wiring until the upstream loopback provider contract is released; track it in the upstream [`lazykatok`](https://github.com/changeroa/lazykatok) repository. |
| mailcrawl | Pending upstream provider contract | No shared-runtime wiring until the upstream loopback provider contract is released; see pending upstream issue #31. |
| qmd | Untouched | qmd retains its own native update, BM25, vector, and query lifecycle. |
| clawgallery | Untouched | ClawGallery retains its VDR/native retrieval lifecycle. |
| Lexical-only crawlers | Unchanged | No semantic provider or embedding configuration is added. |

AutoRAG never overwrites an explicit discrawl `configPath`. It only manages a
config file with the marker above, preserves the native config schema, checks
native metadata for provider/model/dimension changes, and requests the native
rebuild path when needed. A config marker is not ownership of the discrawl
archive or vector store.

## Diagnostics

Use machine-readable output when diagnosing a release or local workspace:

```bash
autorag gateway status --format json
autorag models verify --profile qwen3-embedding-0.6b
autorag status --json
```

Relevant diagnostics include:

- `offline-missing`, `hash-mismatch`, and `download` for cache assets;
- `starting`, `unavailable`, `incompatible`, and `failed` for gateway health;
- `embedder-unavailable` when MinSync cannot reach the local service;
- `embedding-identity-mismatch` when MinSync forces a full reindex;
- `MinSyncQueryError` with a dimension mismatch and reindex guidance when a
  query would reuse an incompatible vector store;
- discrawl's native embedding configuration, metadata, and FTS degradation
  diagnostics.

Diagnostics sanitize paths and secrets where the runtime exposes process
failure details. AutoRAG does not send archive IDs, source paths, credentials,
or vector-store paths to the embedding service: requests contain embedding
text and selected model/profile data only.

## Packaging and release

The npm package includes the gateway JavaScript in `dist/`, the CLI command
bundle, the compliance manifest, and the `licenses/` notices. It does not
include mutable `.gguf`, `.bin`, or `.safetensors` model weights. The release
stager downloads the three pinned llama.cpp platform archives, verifies their
SHA-256 values and required `llama-server` archive member, and publishes those
archives separately from the npm package.

## Release gates

The following are pending human or CI gates and are not claimed by this
scaffolding:

- **Legal sign-off** for the selected Qwen default and the EmbeddingGemma
  compliance bundle, including the Gemma Terms flow-down.
- **Windows x64 native acceptance**, including CPU/Vulkan selection, paths with
  spaces, per-user installation, and orphan-process cleanup.
- **macOS signing and notarization**, including signed nested binaries,
  hardened runtime, quarantine-free launch, and Metal acceptance.

Release jobs must verify pinned hashes and archive members, publish the
compliance bundle, and keep mutable model weights out of the npm artifact.

## Pinned compliance manifest

The following JSON is the documentation copy of
`licenses/embedding-assets.json`. `scripts/check-embedding-manifest.mjs`
compares this block with the machine manifest and with the pinned values in
`src/embedding-runtime/manifest.ts`.

<!-- EMBEDDING-ASSETS-MANIFEST:BEGIN -->
```json
{
  "schemaVersion": 1,
  "generatedFrom": "src/embedding-runtime/manifest.ts",
  "profiles": [
    {
      "profileId": "qwen3-embedding-0.6b",
      "provider": "qwen",
      "model": "Qwen3-Embedding-0.6B-Q8_0.gguf",
      "dimension": 1024,
      "queryPrefix": "",
      "passagePrefix": "",
      "runtimeBuild": "b10951",
      "modelRevision": "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
      "artifactSha256": "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
      "backend": "auto",
      "modelAssetId": "qwen3-embedding-0.6b",
      "licenseId": "Apache-2.0",
      "noticeFile": "licenses/qwen3-embedding-notice.txt",
      "noticeReference": "Qwen3 model card and Apache-2.0 license"
    },
    {
      "profileId": "embeddinggemma-300m",
      "provider": "google",
      "model": "embeddinggemma-300M-Q8_0.gguf",
      "dimension": 768,
      "queryPrefix": "task: search result | query: ",
      "passagePrefix": "title: none | text: ",
      "runtimeBuild": "b10951",
      "modelRevision": "0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
      "artifactSha256": "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
      "backend": "auto",
      "modelAssetId": "embeddinggemma-300m",
      "licenseId": "Gemma Terms of Use",
      "noticeFile": "licenses/gemma-notice.txt",
      "noticeReference": "Google EmbeddingGemma model card and Gemma Terms of Use"
    }
  ],
  "assets": [
    {
      "kind": "model",
      "id": "qwen3-embedding-0.6b",
      "filename": "Qwen3-Embedding-0.6B-Q8_0.gguf",
      "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf",
      "revision": "370f27d7550e0def9b39c1f16d3fbaa13aa67728",
      "sha256": "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
      "licenseId": "Apache-2.0",
      "noticeFile": "licenses/qwen3-embedding-notice.txt",
      "noticeReference": "Qwen3 model card and Apache-2.0 license"
    },
    {
      "kind": "model",
      "id": "embeddinggemma-300m",
      "filename": "embeddinggemma-300M-Q8_0.gguf",
      "url": "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf",
      "revision": "0f741b5a6585bd53aeb15cd1372c56f2a0f65e12",
      "sha256": "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63",
      "licenseId": "Gemma Terms of Use",
      "noticeFile": "licenses/gemma-notice.txt",
      "noticeReference": "Google EmbeddingGemma model card and Gemma Terms of Use"
    },
    {
      "kind": "runtime",
      "id": "llama-macos-arm64",
      "platform": "darwin-arm64-metal",
      "filename": "llama-b10951-bin-macos-arm64.tar.gz",
      "url": "https://github.com/ggml-org/llama.cpp/releases/download/b10951/llama-b10951-bin-macos-arm64.tar.gz",
      "revision": "b10951",
      "sha256": "93d024186f1e6ff1d221f5e0b03567f74dc27a49f5bd42c83066d30af4fbbfec",
      "licenseId": "MIT",
      "noticeFile": "licenses/llama.cpp-MIT.txt",
      "archiveMembers": [
        "llama-b10951/llama-server"
      ]
    },
    {
      "kind": "runtime",
      "id": "llama-win-cpu-x64",
      "platform": "win-x64-cpu",
      "filename": "llama-b10951-bin-win-cpu-x64.zip",
      "url": "https://github.com/ggml-org/llama.cpp/releases/download/b10951/llama-b10951-bin-win-cpu-x64.zip",
      "revision": "b10951",
      "sha256": "ec79f36abd0545ebfad5f61a0c965605058022567659deb21bec711268c3421f",
      "licenseId": "MIT",
      "noticeFile": "licenses/llama.cpp-MIT.txt",
      "archiveMembers": [
        "llama-server.exe"
      ]
    },
    {
      "kind": "runtime",
      "id": "llama-win-vulkan-x64",
      "platform": "win-x64-vulkan",
      "filename": "llama-b10951-bin-win-vulkan-x64.zip",
      "url": "https://github.com/ggml-org/llama.cpp/releases/download/b10951/llama-b10951-bin-win-vulkan-x64.zip",
      "revision": "b10951",
      "sha256": "1e36655f134ab7b94790e3a6311a833dd603c6535b45588ff5b12bdfa79da385",
      "licenseId": "MIT",
      "noticeFile": "licenses/llama.cpp-MIT.txt",
      "archiveMembers": [
        "llama-server.exe"
      ]
    }
  ]
}
```
<!-- EMBEDDING-ASSETS-MANIFEST:END -->
