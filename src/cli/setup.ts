import { accessSync, constants, existsSync } from "node:fs";
import { delimiter, join } from "node:path";
import { resolveAutoRAGHome } from "../config/home.ts";
import { createEmbeddingRuntime, type RuntimeStatus } from "../embedding-runtime/index.ts";
import { resolveProfile } from "../embedding-runtime/manifest.ts";
import type { ProfileId } from "../embedding-runtime/types.ts";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { DEFAULT_MINSYNC_EMBEDDER_DIMENSION, DEFAULT_MINSYNC_EMBEDDER_ID } from "../minsync/embedder-config.ts";
import { readRawConfigObject, writeConfigObject } from "./config.ts";

export type SetupDatasourceState = "configured" | "skipped" | "blocked";
export interface SetupDatasourceReport {
	readonly name: string;
	readonly state: SetupDatasourceState;
	readonly reason?: string;
}
export interface SetupReport {
	readonly ok: boolean;
	readonly mode: "semantic" | "bm25";
	readonly runtime: {
		readonly state: string;
		readonly health: boolean;
		readonly model: string;
		readonly reason?: string;
	};
	readonly model: { readonly valid: boolean; readonly profile: ProfileId };
	readonly datasources: readonly SetupDatasourceReport[];
	readonly remediation?: string;
	readonly changed: boolean;
}

export interface SetupRuntime {
	runtimeStatus(): Promise<RuntimeStatus>;
	verifyModel(profileId?: ProfileId): Promise<{ profileId: ProfileId; path: string; hash: string }>;
	ensureRuntime?(input?: { profileId?: ProfileId }): Promise<{
		baseUrl: string;
		profile: ReturnType<typeof resolveProfile>;
		identity: { provider: string; model: string; dimension: number; profileId: ProfileId };
	}>;
	releaseRuntimeHandles?(): Promise<void>;
}
export interface SetupDeps {
	readonly runtime?: SetupRuntime;
	readonly env?: NodeJS.ProcessEnv;
	readonly pathExists?: (path: string) => boolean;
	readonly executable?: (name: string) => string | undefined;
	readonly acquireLock?: (path: string) => FileLockHandle;
}

const PROFILE: ProfileId = "qwen3-embedding-0.6b";
const LOCK_TIMEOUT_MS = 10_000;
const LOCK_STALE_MS = 30_000;
const REMEDIATION =
	"Semantic embeddings are unavailable. Run `autorag models prefetch` or `autorag models import <file>`, then run `autorag setup` again.";
const BUILTIN_BINARIES: Readonly<Record<string, string>> = {
	kakao: "lazykatok",
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
		.replace(/(?:[A-Za-z]:)?\/[^\s'"`]+/g, "<path>")
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
function storeFor(name: string, entry: Record<string, unknown> | undefined, workspace: string): string | undefined {
	for (const key of ["storePath", "databasePath", "dataDir", "vaultPath", "configPath"]) {
		if (typeof entry?.[key] === "string") return entry[key] as string;
	}
	if (name === "discord") return join(workspace, ".autorag", "datasources", "discrawl", "discrawl.db");
	if (name === "kakao") return join(workspace, ".autorag", "datasources", "lazykatok");
	return undefined;
}
function datasourceNames(config: Record<string, unknown>): string[] {
	const configured = Object.keys((config.datasources ?? {}) as Record<string, unknown>);
	return [
		...new Set([
			...configured,
			"discord",
			"kakao",
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
	profileId?: ProfileId;
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
	let runtime: SetupRuntime | undefined;
	try {
		const raw = readRawConfigObject(options.configPath);
		const configuredProfile = (raw.minSync as Record<string, unknown> | undefined)?.embedder;
		const profileId =
			options.profileId ??
			(configuredProfile &&
			typeof configuredProfile === "object" &&
			typeof (configuredProfile as Record<string, unknown>).profile === "string"
				? ((configuredProfile as Record<string, unknown>).profile as ProfileId)
				: PROFILE);
		runtime =
			options.deps?.runtime ??
			createEmbeddingRuntime({ cacheRoot: home, offline: env.AUTORAG_OFFLINE === "1", fetch: fetch });
		const [runtimeResult, modelResult] = await Promise.allSettled([
			runtime.runtimeStatus(),
			runtime.verifyModel(profileId),
		]);
		const runtimeStatus = runtimeResult.status === "fulfilled" ? runtimeResult.value : undefined;
		const runtimeFailure = runtimeResult.status === "rejected" ? safeReason(runtimeResult.reason) : undefined;
		const modelValid = modelResult.status === "fulfilled";
		const profile = resolveProfile(profileId);
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
			const store = storeFor(type, entry?.connector as Record<string, unknown> | undefined, options.workspacePath);
			if (store && !exists(store)) {
				datasources.push({ name, state: "skipped", reason: "native store is not present" });
				continue;
			}
			if (value === undefined) {
				datasources.push({ name, state: "skipped", reason: "not configured" });
				continue;
			}
			const semanticDatasource = new Set(["discord", "kakao", "slack", "telegram", "whatsapp", "mailcrawl"]);
			if (semanticDatasource.has(type) && !modelValid) {
				datasources.push({ name, state: "blocked", reason: "embedding model is unavailable" });
			} else {
				datasources.push({ name, state: "configured" });
			}
		}
		let changed = false;
		if (modelValid) {
			const minSync = raw.minSync && typeof raw.minSync === "object" ? (raw.minSync as Record<string, unknown>) : {};
			const embedder =
				minSync.embedder && typeof minSync.embedder === "object"
					? (minSync.embedder as Record<string, unknown>)
					: {};
			// An existing embedder is operator-owned, including remote endpoints and
			// dimensions. Only create the supported local profile when absent.
			const nextEmbedder =
				Object.keys(embedder).length > 0
					? embedder
					: options.profileId !== undefined
						? {
								id: profile.model,
								profile: profile.profileId,
								dimension: profile.dimension,
								queryPrefix: profile.queryPrefix,
								passagePrefix: profile.passagePrefix,
							}
						: {
								id: DEFAULT_MINSYNC_EMBEDDER_ID,
								dimension: DEFAULT_MINSYNC_EMBEDDER_DIMENSION,
							};
			const next = {
				...raw,
				minSync: {
					...minSync,
					enabled: minSync.enabled !== false,
					autoInstall: minSync.autoInstall !== false,
					embedder: nextEmbedder,
				},
			};
			if (JSON.stringify(next) !== JSON.stringify(raw)) {
				lock.assertOwned();
				writeConfigObject(options.configPath, next);
				changed = true;
			}
		}
		let effectiveRuntimeStatus = runtimeStatus;
		let startupFailure: string | undefined;
		if (modelValid && effectiveRuntimeStatus?.health.ok !== true && runtime.ensureRuntime !== undefined) {
			try {
				await runtime.ensureRuntime({ profileId });
				effectiveRuntimeStatus = await runtime.runtimeStatus();
			} catch (error) {
				// Runtime startup is independent from model verification and datasource probes.
				// The report remains useful in BM25 mode when startup is unavailable, but the
				// failure reason must reach the caller instead of being discarded.
				startupFailure = safeReason(error);
			}
		}
		const semantic = modelValid && effectiveRuntimeStatus?.health.ok === true;
		const healthFailure =
			effectiveRuntimeStatus !== undefined && effectiveRuntimeStatus.health.ok === false
				? effectiveRuntimeStatus.health.message
				: undefined;
		const runtimeReason = startupFailure ?? runtimeFailure ?? healthFailure;
		return {
			ok: semantic || datasources.some((d) => d.state === "configured"),
			mode: semantic ? "semantic" : "bm25",
			runtime: {
				state: effectiveRuntimeStatus?.state ?? runtimeFailure ?? "unavailable",
				health: effectiveRuntimeStatus?.health.ok === true,
				model: profile.model,
				...(runtimeReason === undefined ? {} : { reason: runtimeReason }),
			},
			model: { valid: modelValid, profile: profile.profileId },
			datasources,
			...(semantic ? {} : { remediation: REMEDIATION }),
			changed,
		};
	} finally {
		lock.release();
		if (options.deps?.runtime === undefined && runtime && typeof runtime.releaseRuntimeHandles === "function") {
			await runtime.releaseRuntimeHandles();
		}
	}
}
