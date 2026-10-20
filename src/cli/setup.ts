import { accessSync, constants, existsSync } from "node:fs";
import { delimiter, isAbsolute, join } from "node:path";
import { resolveAutoRAGHome, resolveUserHome } from "../config/home.ts";
import {
	createEmbedderFromConfig,
	type EmbeddingBackend,
	type EmbeddingIdentity,
	type ManagedEmbedder,
} from "../embedding/index.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { DEFAULT_MINSYNC_EMBEDDER_DIMENSION, DEFAULT_MINSYNC_EMBEDDER_ID } from "../minsync/embedder-config.ts";
import type { MinSyncEmbedderConfig } from "../minsync/types.ts";
import { normalizeEmbedder, readRawConfigObject, writeConfigObject } from "./config.ts";

export type SetupDatasourceState = "configured" | "skipped" | "blocked";
export interface SetupDatasourceReport {
	readonly name: string;
	readonly state: SetupDatasourceState;
	readonly reason?: string;
}
export interface SetupEmbeddingReport {
	/** Where text is embedded: a configured remote endpoint, the `minsync serve` daemon, or MinSync in-process. */
	readonly backend?: Exclude<EmbeddingBackend, "none">;
	readonly provider?: string;
	readonly model?: string;
	readonly dimension?: number;
	/** Why no embedding backend is available. */
	readonly reason?: string;
}
export interface SetupReport {
	readonly ok: boolean;
	readonly mode: "semantic" | "bm25";
	readonly embedding: SetupEmbeddingReport;
	readonly datasources: readonly SetupDatasourceReport[];
	readonly remediation?: string;
	readonly changed: boolean;
}

export interface SetupDeps {
	/** Embedding backend probe; defaults to the one built from the config's `minSync.embedder`. */
	readonly embedder?: ManagedEmbedder;
	readonly env?: NodeJS.ProcessEnv;
	readonly pathExists?: (path: string) => boolean;
	readonly executable?: (name: string) => string | undefined;
	readonly acquireLock?: (path: string) => FileLockHandle;
}

const LOCK_TIMEOUT_MS = 10_000;
const LOCK_STALE_MS = 30_000;
const REMEDIATION =
	"Semantic search needs an embedding backend: run `minsync serve --idle-timeout 0` as a service (docs/minsync-setup.md), or point `minSync.embedder` at a remote endpoint, then run `autorag setup` again.";
const BUILTIN_BINARIES: Readonly<Record<string, string>> = {
	kakao: "lazykatok",
	lark: "lark-cli",
	whatsapp: "wacrawl",
	telegram: "telecrawl",
	slack: "slacrawl",
	discord: "discrawl",
	notion: "notcrawl",
	mailcrawl: "mailcrawl",
	"cloud-drive": "rclone",
	obsidian: "qmd",
};
/**
 * Connectors AutoRAG authenticates itself. Every other built-in datasource is
 * CLI-backed: the external CLI owns its archive, its index, and its
 * credentials, so the probe never requires an env token for one.
 */
const DEFAULT_CREDENTIALS: Readonly<Record<string, readonly string[]>> = {
	github: ["GITHUB_TOKEN"],
};

function executableInPath(name: string, env: NodeJS.ProcessEnv): string | undefined {
	if (name.includes("/") || name.includes("\\")) {
		try {
			accessSync(name, constants.X_OK);
			return name;
		} catch {
			return undefined;
		}
	}
	for (const dir of (env.PATH ?? "").split(delimiter)) {
		if (!dir) continue;
		const candidate = join(dir, process.platform === "win32" ? `${name}.exe` : name);
		try {
			accessSync(candidate, constants.X_OK);
			return candidate;
		} catch {}
	}
	return undefined;
}
function safeReason(value: unknown): string {
	const text = value instanceof Error ? value.message : String(value);
	return text
		.replace(/(?<![\w@.-])(?:[A-Za-z]:)?\/[^\s'"`]+/g, "<path>")
		.replace(/\b(?:token|key|secret|password)\s*[=:]\s*\S+/gi, "$1=<redacted>")
		.slice(0, 180);
}
function configuredCredentialNames(entry: Record<string, unknown> | undefined, datasource: string): string[] {
	const names = [...(DEFAULT_CREDENTIALS[datasource] ?? [])];
	for (const key of ["tokenEnv", "apiKeyEnv", "credentialEnv"]) {
		if (typeof entry?.[key] === "string") names.push(entry[key] as string);
	}
	return [...new Set(names)];
}
/**
 * Native-store locations a datasource may use; the store counts as present
 * when any of them exists. An explicit connector path wins. Discord reads
 * discrawl's own default store (the client never forces an AutoRAG-managed
 * config unless the operator configures one), so every documented discrawl
 * default is accepted, plus the AutoRAG-managed workspace store.
 */
function storesFor(
	name: string,
	entry: Record<string, unknown> | undefined,
	workspace: string,
	env: NodeJS.ProcessEnv,
): readonly string[] {
	for (const key of ["storePath", "databasePath", "dataDir", "vaultPath", "configPath"]) {
		if (typeof entry?.[key] === "string") return [entry[key] as string];
	}
	if (name === "discord") return discrawlStoreCandidates(workspace, env);
	if (name === "kakao") return [join(workspace, ".autorag", "datasources", "lazykatok")];
	return [];
}

function discrawlStoreCandidates(workspace: string, env: NodeJS.ProcessEnv): readonly string[] {
	const home = resolveUserHome(env);
	// XDG_DATA_HOME must be absolute; a relative value is ignored per the spec.
	const xdgDataHome = env.XDG_DATA_HOME;
	return [
		...(xdgDataHome && isAbsolute(xdgDataHome) ? [join(xdgDataHome, "discrawl", "discrawl.db")] : []),
		join(home, "Library", "Application Support", "discrawl", "discrawl.db"),
		join(home, ".local", "share", "discrawl", "discrawl.db"),
		join(home, ".discrawl", "discrawl.db"),
		join(workspace, ".autorag", "datasources", "discrawl", "discrawl.db"),
	];
}
function datasourceNames(config: Record<string, unknown>): string[] {
	const configured = Object.keys((config.datasources ?? {}) as Record<string, unknown>);
	return [
		...new Set([
			...configured,
			"discord",
			"kakao",
			"lark",
			"slack",
			"telegram",
			"whatsapp",
			"mailcrawl",
			"mail-export",
			"notion",
			"github",
			"cloud-drive",
			"obsidian",
			"rss",
			"spotlight",
		]),
	];
}

export async function runSetup(options: {
	configPath: string;
	workspacePath: string;
	deps?: SetupDeps;
}): Promise<SetupReport> {
	const env = options.deps?.env ?? process.env;
	const home = resolveAutoRAGHome(env);
	const lock =
		options.deps?.acquireLock?.(join(home, "setup.lock")) ??
		acquireFileLock(join(home, "setup.lock"), {
			timeoutMs: LOCK_TIMEOUT_MS,
			staleMs: LOCK_STALE_MS,
			retryMs: 10,
			timeoutError: () => new Error("Timed out waiting for setup lock"),
		});
	let embedder: ManagedEmbedder | undefined;
	try {
		const raw = readRawConfigObject(options.configPath);
		const rawMinSync = raw.minSync && typeof raw.minSync === "object" ? (raw.minSync as Record<string, unknown>) : {};
		let configuredEmbedder: MinSyncEmbedderConfig | undefined;
		try {
			configuredEmbedder = normalizeEmbedder(rawMinSync.embedder, "minSync.embedder");
		} catch {
			configuredEmbedder = undefined;
		}
		embedder = options.deps?.embedder ?? createEmbedderFromConfig({ embedder: configuredEmbedder, env });
		let embedding: SetupEmbeddingReport;
		try {
			const health = await embedder.health();
			if (health.alive && health.backend !== "none") {
				const identity: EmbeddingIdentity = await embedder.identity();
				embedding = {
					backend: health.backend,
					provider: identity.provider,
					model: identity.model,
					dimension: identity.dimension,
				};
			} else {
				// The way out is the report's own `remediation` field; the reason is only what was observed.
				embedding = { reason: safeReason(health.detail ?? "no embedding backend answers") };
			}
		} catch (error) {
			// Embedding is independent from datasource probes: the report stays useful in BM25 mode, but the
			// failure reason must reach the caller instead of being discarded.
			embedding = { reason: safeReason(error) };
		}
		const semantic = embedding.backend !== undefined;
		const datasources: SetupDatasourceReport[] = [];
		const source = (raw.datasources ?? {}) as Record<string, unknown>;
		const exists = options.deps?.pathExists ?? existsSync;
		const executable = options.deps?.executable ?? ((name: string) => executableInPath(name, env));
		for (const name of datasourceNames(raw)) {
			const value = source[name];
			if (value === false) {
				datasources.push({ name, state: "skipped", reason: "disabled by operator" });
				continue;
			}
			const entry = value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
			const type = typeof entry?.type === "string" ? entry.type : name;
			const binary = BUILTIN_BINARIES[type];
			if (
				binary &&
				executable(
					typeof entry?.connector === "object" &&
						entry.connector &&
						typeof (entry.connector as Record<string, unknown>).binaryPath === "string"
						? ((entry.connector as Record<string, unknown>).binaryPath as string)
						: binary,
				) === undefined
			) {
				datasources.push({ name, state: "skipped", reason: `missing ${binary} binary` });
				continue;
			}
			const connector = entry?.connector as Record<string, unknown> | undefined;
			// A CLI-backed datasource owns its own credentials (native store,
			// keychain, tool config), so an env credential is never a requirement for
			// it — the probe must mirror what a refresh actually needs.
			if (binary === undefined) {
				const missingCredential = configuredCredentialNames(connector, type).find(
					(key) => env[key] === undefined || env[key] === "",
				);
				if (missingCredential) {
					datasources.push({
						name,
						state: "skipped",
						reason: `credential ${missingCredential} is unavailable`,
					});
					continue;
				}
			}
			const stores = storesFor(
				type,
				entry?.connector as Record<string, unknown> | undefined,
				options.workspacePath,
				env,
			);
			if (stores.length > 0 && !stores.some((store) => exists(store))) {
				datasources.push({ name, state: "skipped", reason: "native store is not present" });
				continue;
			}
			if (value === undefined) {
				datasources.push({ name, state: "skipped", reason: "not configured" });
				continue;
			}
			const semanticDatasource = new Set(["discord", "kakao", "slack", "telegram", "whatsapp", "mailcrawl"]);
			if (semanticDatasource.has(type) && !semantic) {
				datasources.push({ name, state: "blocked", reason: "embedding model is unavailable" });
			} else {
				datasources.push({ name, state: "configured" });
			}
		}
		let changed = false;
		// An existing embedder block is operator-owned, including remote endpoints and dimensions. Only an
		// absent one is created, and it names MinSync's own local default rather than any AutoRAG runtime.
		if (
			semantic &&
			Object.keys(rawMinSync.embedder && typeof rawMinSync.embedder === "object" ? rawMinSync.embedder : {})
				.length === 0
		) {
			const next = {
				...raw,
				minSync: {
					...rawMinSync,
					enabled: true,
					autoInstall: rawMinSync.autoInstall !== false,
					embedder: { id: DEFAULT_MINSYNC_EMBEDDER_ID, dimension: DEFAULT_MINSYNC_EMBEDDER_DIMENSION },
				},
			};
			lock.assertOwned();
			writeConfigObject(options.configPath, next);
			changed = true;
		}
		return {
			ok: semantic || datasources.some((d) => d.state === "configured"),
			mode: semantic ? "semantic" : "bm25",
			embedding,
			datasources,
			...(semantic ? {} : { remediation: REMEDIATION }),
			changed,
		};
	} finally {
		lock.release();
		// A caller-supplied embedder belongs to the caller; one built here is released with the process.
		if (options.deps?.embedder === undefined) await embedder?.close();
	}
}
