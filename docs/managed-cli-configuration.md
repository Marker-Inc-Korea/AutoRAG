# Managed CLI configuration and execution

This is the normative design and contributor guide for CLI-backed
integrations. It defines how AutoRAG transports configuration and enforces
process boundaries without replacing the native command language of an
upstream CLI.

Datasource CLIs and retrieval-owned CLIs use the same low-level provider
contract, but different registries and managers. The datasource registry
contains source connectors such as crawlers, qmd, rclone, and himalaya. The
retrieval runtime contains engines that own local indexes or discovery state:
currently MinSync and optional Jikji. They are not datasource skills and do not
participate in datasource access filtering or datasource result curation.

## Boundary and trust model

The trusted application owns the registry, provider selection, workspace
ownership decision, path checks, secret policy, diagnostics, and process
launch. A provider owns only the upstream-specific configuration transport:
file format, environment variables, flags, profile names, or working
directory. The provider never defines a shared command taxonomy.

```text
trusted config
    -> ManagedCliRegistry
    -> ManagedCliConfigManager
    -> ManagedCliConfigProvider.materialize()
    -> ManagedCliLaunchContext
    -> native CLI + unchanged native args
```

The model may learn native commands from the datasource skill and the CLI's
own `--help`. Datasource retrieval/indexing tools are the approved process
launch surface. Parent `bash` is a filesystem and discovery surface, not a
way to bypass managed datasource execution.

## Provider contract

The shared contract is intentionally small:

```ts
interface ManagedCliConfigProvider {
  readonly tool: string;
  readonly aliases?: readonly string[];
  readonly binaryPaths?: readonly string[];
  readonly managedConfigPath?: (context: ManagedCliContext) => string;
  readonly readConfig?: (path: string) => unknown;
  readonly renderConfig?: (config: unknown, existing: unknown) => string;
  materialize(context: ManagedCliContext): Promise<ManagedCliLaunchContext>;
  inspect(context: ManagedCliContext): Promise<ManagedCliConfigStatus>;
}
```

`ManagedCliLaunchContext` contains `ownership`, optional `cwd`, an environment
map, and `prefixArgs`. `prefixArgs` may transport configuration such as
`--config PATH` or `--workspace PATH`; it must not contain or interpret the
native command. Callers append native command arguments unchanged.

## Ownership

Managed mode is the default. Managed state is workspace-local and provider
specific. The manager writes only provider-declared configuration artifacts,
atomically and with restricted permissions where supported. Existing
unowned fields must be preserved.

External mode requires an explicit trusted absolute path or profile. AutoRAG
passes it through and may inspect it, but never rewrites it. A managed path
and external declaration that conflict fail closed with a stable diagnostic;
they are never silently merged or given implicit precedence.

Providers may use a more specific legacy path during migration, but the path
must be centralized in the provider and documented. New one-off layouts are
not allowed.

## Configuration mechanisms

Providers may use any native mechanism:

| Mechanism | Typical launch context |
| --- | --- |
| Config file | `prefixArgs: ["--config", path]` |
| Environment | `env: { TOOL_CONFIG: path }` |
| Workspace flag | `prefixArgs: ["--workspace", path]` |
| Working directory | `cwd: managedWorkspace` |
| Profile | provider-resolved env/flag reference |

The shared manager must not parse TOML, YAML, JSON, XDG, or an upstream
schema. When a CLI generates its own configuration, its provider can
materialize transport without replacing that native generation flow.

## Workspace and migration

Use stable per-tool and per-instance state, for example:

```text
<workspace>/.autorag/tools/<tool>/<instance>/
  config/
  data/
  cache/
  state/
```

Existing integrations may retain a documented compatibility location while
they migrate. The migration entry must state the old path, new path,
ownership, transport method, preserved settings, and rollback/backward
compatibility behavior. Do not read private crawler or archive database
schemas directly.

## Secret policy

Managed configuration may contain only non-secret settings and references such
as `TOKEN_ENV=MY_SERVICE_TOKEN`, keychain item names, or profile names. It must
never contain secret values, OAuth refresh credentials, cookies, passwords, or
tokens. Secret-looking fields are rejected at the manager boundary. Values
must not appear in logs, diagnostics, argv snapshots, or result metadata.

## Bash direct-execution gate

The registry is the sole source of managed binary names, aliases, and
registered paths. Before shell execution, the parent bash tool rejects
registered CLIs in executable position, including absolute paths, `env` or
other simple wrappers, quoting, chains, pipelines, and subshells. The gate
does not reject ordinary filesystem commands, false-positive names, or
unrelated user binaries.

The stable remediation is:

```text
AUTORAG_MANAGED_CLI_BLOCKED
```

The diagnostic tells the caller to use the datasource tool or the
configuration-enforcing managed execution surface. Prompt instructions are
not a security boundary. The bash tool checks both the datasource registry and
the retrieval registry, while each domain retains its own launch manager.

## Native command access

Datasource skills must document concise native examples and point to
`<binary> --help` for commands not listed. If arbitrary native commands are
needed, a managed pass-through surface may accept `{ tool, args }` after
registry lookup and policy checks. It must apply the same launch context and
pass `args` unchanged. It must not turn native commands into an AutoRAG
`sync/search/doctor` abstraction.

## Current migration matrix

| CLI | Datasource | Transport | Managed state / compatibility |
| --- | --- | --- | --- |
| `discrawl` | Discord | TOML + `--config` | `.autorag/datasources/discrawl/` |
| `katok` | KakaoTalk | `--workspace` | `.autorag/datasources/katok/` |
| `wacrawl` | WhatsApp | `--db` / `--source` | `.autorag/datasources/wacrawl/` |
| `telecrawl` | Telegram | `--db` / `--source` | `.autorag/datasources/telecrawl/` |
| `slacrawl` | Slack | managed database or external `--config` | `.autorag/datasources/slacrawl/` |
| `notcrawl` | Notion | managed database or external `--config` | `.autorag/datasources/notcrawl/` |
| `qmd` | Obsidian | `QMD_CONFIG_DIR` / `XDG_CACHE_HOME` | `.autorag/datasources/obsidian/` |
| `minsync` | retrieval engine, not datasource | managed cwd; native `.minsync` | `.autorag/tools/minsync/` + `.minsync/` |
| `jikji` | retrieval discovery layer, not datasource | managed cwd | `.autorag/tools/jikji/` + root-scoped state |
| `rclone` | Google Drive / remote | `RCLONE_CONFIG` | `.autorag/datasources/rclone/` |
| `himalaya` | Gmail / IMAP | `HIMALAYA_CONFIG` | `.autorag/datasources/himalaya/` |
| `mailcrawl` | Local email archive | `MAILCRAWL_DATA_DIR` | `.autorag/datasources/mailcrawl/<instance>/` |

Each row requires a provider-owned migration test and a deterministic manual
QA run. The provider must preserve native arguments and report non-secret
ownership, paths, applied mechanism, missing requirements, and drift.

## Contributor checklist

- Register the binary, aliases, and explicit paths.
- Add a provider with ownership and secret policy.
- Route every spawn through materialization.
- Preserve unknown settings and external files.
- Add tests for ownership conflict, paths, secrets, aliases, bypass forms,
  native argument passthrough, diagnostics, and migration.
- Add skill instructions and `--help` guidance.
- Add documentation and manual QA.
- Run focused tests, typecheck, lint, build, and the complete suite.
