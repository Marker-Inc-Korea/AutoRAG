/**
 * `autorag setup` support for the Nogon datasource: is Nogon installed and running, and does it embed with the
 * same model as the rest of AutoRAG?
 *
 * Detection never writes anything. Writing Nogon's config and restarting its service only happen when the operator
 * passes `--sync-nogon`, because that file and that login item belong to Nogon and its operator. Nogon is never
 * modified; this only edits its documented config.yml, keeps a backup first, and restarts the service the same way
 * the operator would.
 */

import { spawn } from "node:child_process";
import { copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import type { EmbedderEndpoint, ManagedEmbedder } from "../../../embedding/embedder.ts";
import type { ResolvedEmbeddingEndpoint } from "../../embedding-env.ts";
import { canonicalDatasourceTemplate } from "../../template-aliases.ts";
import { NogonClient } from "./client.ts";
import {
	applyNogonSemanticSettings,
	deriveNogonSemanticSettings,
	differingSemanticKeys,
	type NogonSemanticKey,
	readNogonSemanticSettings,
	reindexRequired,
} from "./embedding.ts";

export interface NogonSetupEmbedding {
	/** Nogon's config already matches the shared embedding backend. */
	readonly inSync: boolean;
	/** `semantic_search` keys that differ from the shared backend, sorted. */
	readonly differs: readonly NogonSemanticKey[];
	/** Nogon's config file was rewritten by this run. */
	readonly applied: boolean;
	/** Nogon must be restarted to read the new config (it was not, or could not be). */
	readonly restartRequired?: boolean;
	/** Existing pages must be re-embedded (`nogon reindex`) and that was not done. */
	readonly reindexRequired?: boolean;
	readonly reason?: string;
}

export interface NogonSetupReport {
	readonly datasource: string;
	readonly installed: boolean;
	readonly running?: boolean;
	readonly configPath?: string;
	readonly embedding?: NogonSetupEmbedding;
	readonly reason?: string;
	/** What the operator (or an agent) should run next. */
	readonly remediation?: string;
}

export interface NogonServiceDeps {
	/** Nogon's config.yml path, or undefined when it cannot be determined. */
	readonly configPath?: () => string | undefined;
	/** `GET /api/config` against the datasource's baseUrl. */
	readonly health?: (baseUrl: string) => Promise<{
		readonly reachable: boolean;
		readonly semanticEnabled?: boolean;
		readonly reason?: string;
	}>;
	/** Reload a running Nogon server's config (`nogon reload`, v0.21.0+). Preferred over restarting the service. */
	readonly reload?: (baseUrl: string) => Promise<{ readonly ok: boolean; readonly reason?: string }>;
	/** Restart the Nogon service. Omitted means the operator must do it. */
	readonly restart?: () => Promise<{ readonly ok: boolean; readonly reason?: string }>;
	/** `nogon reindex` against the running server. Omitted means the operator must do it. */
	readonly reindex?: (baseUrl: string) => Promise<{ readonly ok: boolean; readonly reason?: string }>;
	readonly now?: () => number;
	/** Pause between polls while a restarted server comes up. Tests replace it. */
	readonly sleep?: (ms: number) => Promise<void>;
	/** How long to wait for a restarted server before giving up on the reindex. Default 30s. */
	readonly waitMs?: number;
}

const INSTALL = "brew install nogon   (other platforms: https://github.com/asciimoo/nogon/releases)";
/** Homebrew names its service after the formula; a not-yet-upgraded install still has the old `hister` one. */
const startCommand = (service: string): string => `brew services start ${service}`;
const restartCommand = (service: string): string => `brew services restart ${service}`;
const SHARE_DAEMON =
	"Nogon can only share a local embedding server. Start `inl serve --idle-timeout 0` (docs/indexing/embedding-backends.md), then run `autorag setup --sync-nogon`.";

interface NogonEntry {
	readonly name: string;
	readonly baseUrl: string;
	readonly semantic: boolean;
	readonly tokenEnv?: string;
}

/**
 * The Homebrew service that runs `binary`. `hister` only for a program still named that (Nogon before v0.22.0, until
 * Homebrew relinks it); everything else, including an unknown binary, is `nogon`.
 */
export function nogonServiceName(binary: string | undefined): string {
	if (binary === undefined) return "nogon";
	const name = basename(binary.replace(/\\/gu, "/")).replace(/\.exe$/iu, "");
	return name === "hister" ? "hister" : "nogon";
}

/** The first configured `nogon` datasource (by template `type` or by name), with its connector options. */
export function findNogonEntries(raw: Record<string, unknown>): readonly NogonEntry[] {
	const datasources = (raw.datasources ?? {}) as Record<string, unknown>;
	const out: NogonEntry[] = [];
	for (const [name, value] of Object.entries(datasources)) {
		if (value === false) continue;
		const entry = value !== null && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
		if (canonicalDatasourceTemplate(typeof entry?.type === "string" ? entry.type : name) !== "nogon") continue;
		const connector =
			entry?.connector !== null && typeof entry?.connector === "object"
				? (entry.connector as Record<string, unknown>)
				: {};
		out.push({
			name,
			baseUrl: typeof connector.baseUrl === "string" ? connector.baseUrl : "http://127.0.0.1:4433",
			semantic: connector.semantic === true,
			...(typeof connector.tokenEnv === "string" ? { tokenEnv: connector.tokenEnv } : {}),
		});
	}
	return out;
}

/**
 * Where Nogon reads its config on this platform, in the order Nogon itself searches: `NOGON_CONFIG`, then the legacy
 * `HISTER_CONFIG`, then for each of the `nogon` and (before v0.22.0) `hister` names the platform's config directories.
 * An upgraded install keeps its `hister` directory, and Nogon reads it until a `nogon` one exists.
 */
export function defaultNogonConfigPath(
	env: NodeJS.ProcessEnv,
	exists: (path: string) => boolean = existsSync,
	platform: NodeJS.Platform = process.platform,
	home: string = env.HOME ?? homedir(),
): string | undefined {
	const xdg = env.XDG_CONFIG_HOME;
	const candidates: string[] = [];
	if (env.NOGON_CONFIG) candidates.push(env.NOGON_CONFIG);
	if (env.HISTER_CONFIG) candidates.push(env.HISTER_CONFIG);
	for (const name of ["nogon", "hister"]) {
		if (platform === "darwin") {
			candidates.push(join(xdg || join(home, "Library", "Preferences"), name, "config.yml"));
			candidates.push(join(home, "Library", "Application Support", name, "config.yml"));
		} else if (platform === "win32") {
			if (env.LOCALAPPDATA) candidates.push(join(env.LOCALAPPDATA, name, "config.yml"));
			if (xdg) candidates.push(join(xdg, name, "config.yml"));
			if (env.APPDATA) candidates.push(join(env.APPDATA, name, "config.yml"));
		} else {
			candidates.push(join(xdg || join(home, ".config"), name, "config.yml"));
		}
		if (name === "hister") candidates.push(join(home, ".histerrc"));
		if (platform === "darwin" || platform === "win32") candidates.push(join(home, ".config", name, "config.yml"));
	}
	return candidates.find((path) => exists(path));
}

function commandRun(
	command: string,
	args: readonly string[],
	timeoutMs: number,
): Promise<{ ok: boolean; reason?: string }> {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	return new Promise((resolve) => {
		let stderr = "";
		const child = spawn(command, args, { stdio: ["ignore", "ignore", "pipe"] });
		const timer = setTimeout(() => {
			child.kill("SIGTERM");
			resolve({ ok: false, reason: `${command} ${args.join(" ")} timed out after ${timeoutMs}ms` });
		}, timeoutMs);
		child.stderr?.on("data", (chunk: Buffer) => {
			stderr = (stderr + chunk.toString()).slice(-600);
		});
		child.on("error", (error) => {
			clearTimeout(timer);
			resolve({ ok: false, reason: error.message });
		});
		child.on("close", (code) => {
			clearTimeout(timer);
			resolve(
				code === 0
					? { ok: true }
					: { ok: false, reason: `${command} ${args.join(" ")} exited ${code}: ${stderr.trim()}` },
			);
		});
	});
}

/** The real service deps. Restart is only offered where a service manager is known: Homebrew on macOS/Linux. */
export function systemNogonDeps(env: NodeJS.ProcessEnv, nogon: string | undefined): NogonServiceDeps {
	return {
		configPath: () => defaultNogonConfigPath(env),
		health: async (baseUrl) => {
			const client = new NogonClient({ baseUrl, env, timeoutMs: 3_000 });
			const result = await client.config();
			if (!result.ok) return { reachable: false, reason: result.message };
			return { reachable: true, semanticEnabled: result.value.semanticEnabled === true };
		},
		...(nogon === undefined
			? {}
			: {
					reload: (baseUrl: string) =>
						commandRun(nogon, ["--server-url", baseUrl, "--client-timeout", "30", "reload"], 35_000),
					restart: () => commandRun("brew", ["services", "restart", nogonServiceName(nogon)], 60_000),
					reindex: (baseUrl: string) =>
						commandRun(nogon, ["--server-url", baseUrl, "--client-timeout", "0", "reindex"], 3_600_000),
				}),
	};
}

function backupPath(configPath: string, now: number): string {
	return join(dirname(configPath), `config.yml.bak.${now}`);
}

async function endpointOf(embedder: ManagedEmbedder): Promise<ResolvedEmbeddingEndpoint | undefined> {
	const endpoint: EmbedderEndpoint | undefined = await embedder.describeEndpoint();
	if (endpoint === undefined) return undefined;
	const dimension = endpoint.dimension ?? (await embedder.identity()).dimension;
	return { ...endpoint, dimension };
}

/**
 * Probe Nogon for every configured `nogon` datasource. Writes only when `sync` is true, and then only Nogon's own
 * config.yml (after a backup), followed by the service restart and reindex the change needs.
 */
export async function setupNogon(options: {
	readonly entries: readonly NogonEntry[];
	readonly embedder: ManagedEmbedder;
	readonly executable: (name: string) => string | undefined;
	readonly deps: NogonServiceDeps;
	readonly sync: boolean;
	readonly assertLockOwned?: () => void;
}): Promise<{ readonly reports: readonly NogonSetupReport[]; readonly changed: boolean }> {
	const reports: NogonSetupReport[] = [];
	let changed = false;
	const binary = options.executable("nogon");
	const service = nogonServiceName(binary);
	for (const entry of options.entries) {
		if (binary === undefined) {
			reports.push({
				datasource: entry.name,
				installed: false,
				reason: "the nogon program is not installed",
				remediation: `Install it: ${INSTALL}. Then ${startCommand("nogon")} and install the Nogon browser extension.`,
			});
			continue;
		}
		const health = await (options.deps.health?.(entry.baseUrl) ?? Promise.resolve({ reachable: true }));
		const configPath = options.deps.configPath?.();
		const base: NogonSetupReport = {
			datasource: entry.name,
			installed: true,
			running: health.reachable,
			...(configPath === undefined ? {} : { configPath }),
			...(health.reachable ? {} : { reason: health.reason ?? "the Nogon server did not answer" }),
		};
		if (!entry.semantic) {
			reports.push(health.reachable ? base : { ...base, remediation: `Start it: ${startCommand(service)}` });
			continue;
		}
		const embedding = await syncEmbedding({ ...options, entry, configPath, health, binary });
		changed ||= embedding.applied;
		const remediation: string[] = [];
		if (!health.reachable) remediation.push(`Start it: ${startCommand(service)}`);
		if (!embedding.inSync && !embedding.applied && !options.sync && embedding.reason === undefined) {
			remediation.push("Make Nogon embed with the shared model: autorag setup --sync-nogon");
		}
		if (embedding.reason?.includes("inl serve"))
			remediation.push(
				SHARE_DAEMON.replace(
					"Nogon can only share a local embedding server. ",
					"Start `inl serve --idle-timeout 0` (docs/indexing/embedding-backends.md), then run `autorag setup --sync-nogon`.",
				),
			);
		if (embedding.restartRequired) remediation.push(`Restart Nogon to read its config: ${restartCommand(service)}`);
		if (embedding.reindexRequired) remediation.push("Re-embed existing pages: nogon reindex");
		reports.push({
			...base,
			embedding,
			...(remediation.length === 0 ? {} : { remediation: [...new Set(remediation)].join(" ") }),
		});
	}
	return { reports, changed };
}

async function syncEmbedding(args: {
	readonly entry: NogonEntry;
	readonly embedder: ManagedEmbedder;
	readonly deps: NogonServiceDeps;
	readonly sync: boolean;
	readonly configPath: string | undefined;
	readonly health: { readonly reachable: boolean };
	readonly binary: string;
	readonly assertLockOwned?: () => void;
}): Promise<NogonSetupEmbedding> {
	const { configPath } = args;
	if (configPath === undefined) {
		return {
			inSync: false,
			differs: [],
			applied: false,
			reason: "Nogon's config.yml was not found; start Nogon once or set NOGON_CONFIG, then run setup again",
		};
	}
	let text: string;
	try {
		text = readFileSync(configPath, "utf8");
	} catch (error) {
		return {
			inSync: false,
			differs: [],
			applied: false,
			reason: `cannot read ${configPath}: ${(error as Error).message}`,
		};
	}
	const current = readNogonSemanticSettings(text);
	if (!current.ok) return { inSync: false, differs: [], applied: false, reason: `${configPath}: ${current.reason}` };
	const endpoint = await endpointOf(args.embedder);
	if (endpoint === undefined) {
		return {
			inSync: false,
			differs: [],
			applied: false,
			reason:
				"AutoRAG embeds in-process, which Nogon cannot reach: it needs the `inl serve` daemon to share the model",
		};
	}
	const derived = deriveNogonSemanticSettings(endpoint);
	if (!derived.ok) return { inSync: false, differs: [], applied: false, reason: derived.reason };
	const differs = differingSemanticKeys(current.settings, derived.settings);
	if (differs.length === 0) return { inSync: true, differs, applied: false };
	if (!args.sync) return { inSync: false, differs, applied: false };

	const next = applyNogonSemanticSettings(text, derived.settings);
	if (!next.ok) return { inSync: false, differs, applied: false, reason: `${configPath}: ${next.reason}` };
	args.assertLockOwned?.();
	try {
		copyFileSync(configPath, backupPath(configPath, (args.deps.now ?? Date.now)()));
		writeFileSync(configPath, next.text);
	} catch (error) {
		return {
			inSync: false,
			differs,
			applied: false,
			reason: `cannot write ${configPath}: ${(error as Error).message}`,
		};
	}
	const needsReindex = reindexRequired(differs);
	let restartRequired = true;
	let reindexNeeded = needsReindex;
	let reason: string | undefined;
	let reloadedInPlace = false;
	if (args.health.reachable && args.deps.reload !== undefined) {
		const reloaded = await args.deps.reload(args.entry.baseUrl);
		if (reloaded.ok) {
			restartRequired = false;
			reloadedInPlace = true;
		} else if (args.deps.restart === undefined) {
			reason = `Nogon's config was updated but reload failed: ${reloaded.reason ?? "unknown error"}`;
		}
	}
	if (restartRequired && args.deps.restart !== undefined) {
		const restarted = await args.deps.restart();
		if (restarted.ok) restartRequired = false;
		else reason = `Nogon's config was updated but the restart failed: ${restarted.reason ?? "unknown error"}`;
	}
	if (!restartRequired && needsReindex && args.deps.reindex !== undefined) {
		const up = reloadedInPlace ? { ok: true as const } : await waitForServer(args.entry.baseUrl, args.deps);
		if (!up.ok) {
			reason = `Nogon was restarted but did not answer at ${args.entry.baseUrl} within ${up.waitedMs}ms: ${up.reason}`;
		} else {
			const reindexed = await args.deps.reindex(args.entry.baseUrl);
			if (reindexed.ok) reindexNeeded = false;
			else reason = `Nogon config was applied but reindexing failed: ${reindexed.reason ?? "unknown error"}`;
		}
	}
	return {
		inSync: true,
		differs,
		applied: true,
		restartRequired,
		reindexRequired: reindexNeeded,
		...(reason === undefined ? {} : { reason }),
	};
}

const DEFAULT_WAIT_MS = 30_000;
const POLL_MS = 500;

/** `brew services restart` returns before the server listens, so poll until it answers (or the wait runs out). */
async function waitForServer(
	baseUrl: string,
	deps: NogonServiceDeps,
): Promise<{ readonly ok: true } | { readonly ok: false; readonly reason: string; readonly waitedMs: number }> {
	if (deps.health === undefined) return { ok: true };
	const limit = deps.waitMs ?? DEFAULT_WAIT_MS;
	const sleep = deps.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
	let waited = 0;
	let reason = "no answer";
	for (;;) {
		const health = await deps.health(baseUrl);
		if (health.reachable) return { ok: true };
		reason = health.reason ?? reason;
		if (waited >= limit) return { ok: false, reason, waitedMs: waited };
		await sleep(POLL_MS);
		waited += POLL_MS;
	}
}
