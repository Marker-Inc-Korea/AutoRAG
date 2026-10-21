/**
 * `autorag setup` support for the Hister datasource: is Hister installed and running, and does it embed with the
 * same model as the rest of AutoRAG?
 *
 * Detection never writes anything. Writing Hister's config and restarting its service only happen when the operator
 * passes `--sync-hister`, because that file and that login item belong to Hister and its operator. Hister is never
 * modified; this only edits its documented config.yml, keeps a backup first, and restarts the service the same way
 * the operator would.
 */

import { spawn } from "node:child_process";
import { copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { EmbedderEndpoint, ManagedEmbedder } from "../../../embedding/embedder.ts";
import type { ResolvedEmbeddingEndpoint } from "../../embedding-env.ts";
import { HisterClient } from "./client.ts";
import {
	applyHisterSemanticSettings,
	deriveHisterSemanticSettings,
	differingSemanticKeys,
	type HisterSemanticKey,
	readHisterSemanticSettings,
	reindexRequired,
} from "./embedding.ts";

export interface HisterSetupEmbedding {
	/** Hister's config already matches the shared embedding backend. */
	readonly inSync: boolean;
	/** `semantic_search` keys that differ from the shared backend, sorted. */
	readonly differs: readonly HisterSemanticKey[];
	/** Hister's config file was rewritten by this run. */
	readonly applied: boolean;
	/** Hister must be restarted to read the new config (it was not, or could not be). */
	readonly restartRequired?: boolean;
	/** Existing pages must be re-embedded (`hister reindex`) and that was not done. */
	readonly reindexRequired?: boolean;
	readonly reason?: string;
}

export interface HisterSetupReport {
	readonly datasource: string;
	readonly installed: boolean;
	readonly running?: boolean;
	readonly configPath?: string;
	readonly embedding?: HisterSetupEmbedding;
	readonly reason?: string;
	/** What the operator (or an agent) should run next. */
	readonly remediation?: string;
}

export interface HisterServiceDeps {
	/** Hister's config.yml path, or undefined when it cannot be determined. */
	readonly configPath?: () => string | undefined;
	/** `GET /api/config` against the datasource's baseUrl. */
	readonly health?: (baseUrl: string) => Promise<{
		readonly reachable: boolean;
		readonly semanticEnabled?: boolean;
		readonly reason?: string;
	}>;
	/** Restart the Hister service. Omitted means the operator must do it. */
	readonly restart?: () => Promise<{ readonly ok: boolean; readonly reason?: string }>;
	/** `hister reindex` against the running server. Omitted means the operator must do it. */
	readonly reindex?: (baseUrl: string) => Promise<{ readonly ok: boolean; readonly reason?: string }>;
	readonly now?: () => number;
	/** Pause between polls while a restarted server comes up. Tests replace it. */
	readonly sleep?: (ms: number) => Promise<void>;
	/** How long to wait for a restarted server before giving up on the reindex. Default 30s. */
	readonly waitMs?: number;
}

const INSTALL = "brew install hister   (other platforms: https://github.com/asciimoo/hister/releases)";
const START = "brew services start hister";
const SHARE_DAEMON =
	"Hister can only share a local embedding server. Start `minsync serve --idle-timeout 0` (docs/minsync-setup.md), then run `autorag setup --sync-hister`.";

interface HisterEntry {
	readonly name: string;
	readonly baseUrl: string;
	readonly semantic: boolean;
	readonly tokenEnv?: string;
}

/** The first configured `hister` datasource (by template `type` or by name), with its connector options. */
export function findHisterEntries(raw: Record<string, unknown>): readonly HisterEntry[] {
	const datasources = (raw.datasources ?? {}) as Record<string, unknown>;
	const out: HisterEntry[] = [];
	for (const [name, value] of Object.entries(datasources)) {
		if (value === false) continue;
		const entry = value !== null && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
		if ((typeof entry?.type === "string" ? entry.type : name) !== "hister") continue;
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

/** Where Hister reads its config on this platform, in the order Hister itself searches (docs/configuration). */
export function defaultHisterConfigPath(
	env: NodeJS.ProcessEnv,
	exists: (path: string) => boolean = existsSync,
	platform: NodeJS.Platform = process.platform,
	home: string = env.HOME ?? homedir(),
): string | undefined {
	const xdg = env.XDG_CONFIG_HOME;
	const candidates: string[] = [];
	if (env.HISTER_CONFIG) candidates.push(env.HISTER_CONFIG);
	if (platform === "darwin") {
		candidates.push(join(xdg || join(home, "Library", "Preferences"), "hister", "config.yml"));
		candidates.push(join(home, "Library", "Application Support", "hister", "config.yml"));
	} else if (platform === "win32") {
		if (env.LOCALAPPDATA) candidates.push(join(env.LOCALAPPDATA, "hister", "config.yml"));
		if (xdg) candidates.push(join(xdg, "hister", "config.yml"));
		if (env.APPDATA) candidates.push(join(env.APPDATA, "hister", "config.yml"));
	} else {
		candidates.push(join(xdg || join(home, ".config"), "hister", "config.yml"));
	}
	candidates.push(join(home, ".config", "hister", "config.yml"));
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
export function systemHisterDeps(env: NodeJS.ProcessEnv, hister: string | undefined): HisterServiceDeps {
	return {
		configPath: () => defaultHisterConfigPath(env),
		health: async (baseUrl) => {
			const client = new HisterClient({ baseUrl, env, timeoutMs: 3_000 });
			const result = await client.config();
			if (!result.ok) return { reachable: false, reason: result.message };
			return { reachable: true, semanticEnabled: result.value.semanticEnabled === true };
		},
		...(hister === undefined
			? {}
			: {
					restart: () => commandRun("brew", ["services", "restart", "hister"], 60_000),
					reindex: (baseUrl: string) =>
						commandRun(hister, ["--server-url", baseUrl, "--client-timeout", "0", "reindex"], 3_600_000),
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
 * Probe Hister for every configured `hister` datasource. Writes only when `sync` is true, and then only Hister's own
 * config.yml (after a backup), followed by the service restart and reindex the change needs.
 */
export async function setupHister(options: {
	readonly entries: readonly HisterEntry[];
	readonly embedder: ManagedEmbedder;
	readonly executable: (name: string) => string | undefined;
	readonly deps: HisterServiceDeps;
	readonly sync: boolean;
	readonly assertLockOwned?: () => void;
}): Promise<{ readonly reports: readonly HisterSetupReport[]; readonly changed: boolean }> {
	const reports: HisterSetupReport[] = [];
	let changed = false;
	const binary = options.executable("hister");
	for (const entry of options.entries) {
		if (binary === undefined) {
			reports.push({
				datasource: entry.name,
				installed: false,
				reason: "the hister program is not installed",
				remediation: `Install it: ${INSTALL}. Then ${START} and install the Hister browser extension.`,
			});
			continue;
		}
		const health = await (options.deps.health?.(entry.baseUrl) ?? Promise.resolve({ reachable: true }));
		const configPath = options.deps.configPath?.();
		const base: HisterSetupReport = {
			datasource: entry.name,
			installed: true,
			running: health.reachable,
			...(configPath === undefined ? {} : { configPath }),
			...(health.reachable ? {} : { reason: health.reason ?? "the Hister server did not answer" }),
		};
		if (!entry.semantic) {
			reports.push(health.reachable ? base : { ...base, remediation: `Start it: ${START}` });
			continue;
		}
		const embedding = await syncEmbedding({ ...options, entry, configPath, health, binary });
		changed ||= embedding.applied;
		const remediation: string[] = [];
		if (!health.reachable) remediation.push(`Start it: ${START}`);
		if (!embedding.inSync && !embedding.applied && !options.sync && embedding.reason === undefined) {
			remediation.push("Make Hister embed with the shared model: autorag setup --sync-hister");
		}
		if (embedding.reason?.includes("minsync serve"))
			remediation.push(
				SHARE_DAEMON.replace(
					"Hister can only share a local embedding server. ",
					"Start `minsync serve --idle-timeout 0` (docs/minsync-setup.md), then run `autorag setup --sync-hister`.",
				),
			);
		if (embedding.restartRequired)
			remediation.push(`Restart Hister to read its config: ${START.replace("start", "restart")}`);
		if (embedding.reindexRequired) remediation.push("Re-embed existing pages: hister reindex");
		reports.push({
			...base,
			embedding,
			...(remediation.length === 0 ? {} : { remediation: [...new Set(remediation)].join(" ") }),
		});
	}
	return { reports, changed };
}

async function syncEmbedding(args: {
	readonly entry: HisterEntry;
	readonly embedder: ManagedEmbedder;
	readonly deps: HisterServiceDeps;
	readonly sync: boolean;
	readonly configPath: string | undefined;
	readonly health: { readonly reachable: boolean };
	readonly binary: string;
	readonly assertLockOwned?: () => void;
}): Promise<HisterSetupEmbedding> {
	const { configPath } = args;
	if (configPath === undefined) {
		return {
			inSync: false,
			differs: [],
			applied: false,
			reason: "Hister's config.yml was not found; start Hister once or set HISTER_CONFIG, then run setup again",
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
	const current = readHisterSemanticSettings(text);
	if (!current.ok) return { inSync: false, differs: [], applied: false, reason: `${configPath}: ${current.reason}` };
	const endpoint = await endpointOf(args.embedder);
	if (endpoint === undefined) {
		return {
			inSync: false,
			differs: [],
			applied: false,
			reason:
				"AutoRAG embeds in-process, which Hister cannot reach: it needs the `minsync serve` daemon to share the model",
		};
	}
	const derived = deriveHisterSemanticSettings(endpoint);
	if (!derived.ok) return { inSync: false, differs: [], applied: false, reason: derived.reason };
	const differs = differingSemanticKeys(current.settings, derived.settings);
	if (differs.length === 0) return { inSync: true, differs, applied: false };
	if (!args.sync) return { inSync: false, differs, applied: false };

	const next = applyHisterSemanticSettings(text, derived.settings);
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
	if (args.deps.restart !== undefined) {
		const restarted = await args.deps.restart();
		if (restarted.ok) restartRequired = false;
		else reason = `Hister's config was updated but the restart failed: ${restarted.reason ?? "unknown error"}`;
	}
	if (!restartRequired && needsReindex && args.deps.reindex !== undefined) {
		const up = await waitForServer(args.entry.baseUrl, args.deps);
		if (!up.ok) {
			reason = `Hister was restarted but did not answer at ${args.entry.baseUrl} within ${up.waitedMs}ms: ${up.reason}`;
		} else {
			const reindexed = await args.deps.reindex(args.entry.baseUrl);
			if (reindexed.ok) reindexNeeded = false;
			else reason = `Hister was restarted but reindexing failed: ${reindexed.reason ?? "unknown error"}`;
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
	deps: HisterServiceDeps,
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
