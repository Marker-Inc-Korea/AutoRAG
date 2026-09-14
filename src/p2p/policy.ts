import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { parse as parseToml } from "smol-toml";
import { resolveAutoRAGHome } from "../config/home.ts";

export type PolicyTier = "private" | "never" | "always" | "peers";

export interface PolicyEntry {
	readonly tier: PolicyTier;
	readonly peers?: readonly string[];
}

export interface PolicyResolution {
	readonly tier: PolicyTier;
	readonly allowed: boolean;
	readonly shareBytes: boolean;
	readonly redact: boolean;
}

export interface PolicyQuotas {
	readonly queriesPerHour: number;
	readonly burst: number;
	readonly maxBodyBytes: number;
	readonly maxFileBytes: number;
}

export interface PolicyStoreOptions {
	readonly workspacePath: string;
	/** A home directory or an AutoRAG home directory (`~/.autorag`). */
	readonly homePath?: string;
	readonly globalConfigPath?: string;
	readonly workspacePolicyPath?: string;
	readonly seenSourcesPath?: string;
	/** Overrides p2p.newFilesPublic, primarily useful to callers with normalized config. */
	readonly newFilesPublic?: boolean;
}

export class PolicyError extends Error {
	readonly cause?: unknown;

	constructor(message: string, options?: { readonly cause?: unknown }) {
		super(message);
		this.name = "PolicyError";
		this.cause = options?.cause;
	}
}

const DEFAULT_QUOTAS: PolicyQuotas = {
	queriesPerHour: 10,
	burst: 3,
	maxBodyBytes: 65_536,
	maxFileBytes: 26_214_400,
};

const POLICY_TIERS = new Set<PolicyTier>(["private", "never", "always", "peers"]);
const UNSAFE_POLICY_CHARACTERS = /[\p{Cc}\p{Cf}]/u;
const WINDOWS_ABSOLUTE_PATH = /^[a-z]:[\\/]/iu;
const POSIX_FILESYSTEM_ROOTS = new Set([
	"applications",
	"bin",
	"dev",
	"etc",
	"home",
	"library",
	"media",
	"mnt",
	"opt",
	"private",
	"proc",
	"root",
	"run",
	"users",
	"sbin",
	"srv",
	"system",
	"tmp",
	"usr",
	"var",
	"volumes",
]);

interface PolicyMatch {
	readonly pattern: string;
	readonly entry: PolicyEntry;
	readonly order: number;
}

interface SeenSourceState {
	readonly promoted: Set<string>;
	readonly pending: Set<string>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isPositiveInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function normalizeFingerprint(value: unknown, context: string): string {
	if (typeof value !== "string" || value.length === 0 || UNSAFE_POLICY_CHARACTERS.test(value)) {
		throw new PolicyError(`${context} must be a non-empty string without control characters`);
	}
	return value.normalize("NFC");
}

function looksLikeAbsoluteFilesystemPath(key: string, workspacePath: string): boolean {
	if (WINDOWS_ABSOLUTE_PATH.test(key) || key.startsWith("\\") || key.startsWith("//")) return true;
	const normalizedWorkspace = resolve(workspacePath).replaceAll("\\", "/").replace(/\/+$/u, "");
	if (key === normalizedWorkspace || key.startsWith(`${normalizedWorkspace}/`)) return true;
	if (key.startsWith("/")) {
		const firstSegment = key.slice(1).split("/", 1)[0]?.toLocaleLowerCase("en-US");
		if (firstSegment !== undefined && POSIX_FILESYSTEM_ROOTS.has(firstSegment)) return true;
	}
	return false;
}

function validatePolicyKey(key: unknown, workspacePath: string, context: string): string {
	if (typeof key !== "string" || key.length === 0) {
		throw new PolicyError(`${context} must be a non-empty source-identifier glob`);
	}
	const normalized = key.normalize("NFC");
	if (UNSAFE_POLICY_CHARACTERS.test(normalized)) {
		throw new PolicyError(`${context} contains a control character`);
	}
	if (normalized.includes("\\") || normalized.includes("\0")) {
		throw new PolicyError(`${context} must use source-identifier separators, not filesystem separators`);
	}
	const sourceSegments = normalized.startsWith("/") ? normalized.slice(1).split("/") : [];
	if (sourceSegments.some((segment) => segment === "." || segment === "..")) {
		throw new PolicyError(`${context} must not contain path traversal segments`);
	}
	if (!normalized.startsWith("/") && !WINDOWS_ABSOLUTE_PATH.test(normalized)) {
		throw new PolicyError(`${context} must be an absolute local path or slash-prefixed datasource source`);
	}
	if (normalized.startsWith("/") && normalized === "/") {
		throw new PolicyError(`${context} must identify a source namespace`);
	}
	if (/^file:/iu.test(normalized)) {
		throw new PolicyError(`${context} must not use the file scheme`);
	}
	return normalized;
}

function normalizePolicyEntry(value: unknown, context: string): PolicyEntry {
	if (typeof value === "string") {
		if (!POLICY_TIERS.has(value as PolicyTier)) {
			throw new PolicyError(`${context}.tier must be private, never, always, or peers`);
		}
		return { tier: value as PolicyTier };
	}
	if (!isRecord(value)) {
		throw new PolicyError(`${context} must be a policy entry`);
	}

	for (const key of Object.keys(value)) {
		if (key !== "tier" && key !== "peers") {
			throw new PolicyError(`${context} contains unknown field ${JSON.stringify(key)}`);
		}
	}
	const tier = value.tier;
	if (typeof tier !== "string" || !POLICY_TIERS.has(tier as PolicyTier)) {
		throw new PolicyError(`${context}.tier must be private, never, always, or peers`);
	}
	if (value.peers === undefined) return { tier: tier as PolicyTier };
	if (!Array.isArray(value.peers)) {
		throw new PolicyError(`${context}.peers must be an array of fingerprints`);
	}
	const peers = value.peers.map((peer, index) => normalizeFingerprint(peer, `${context}.peers[${index}]`));
	return { tier: tier as PolicyTier, peers };
}

function normalizePolicyEntries(value: unknown, workspacePath: string, context: string): Map<string, PolicyEntry> {
	if (!isRecord(value)) {
		throw new PolicyError(`${context} must be an object keyed by source-identifier globs`);
	}
	const entries = new Map<string, PolicyEntry>();
	for (const [key, entry] of Object.entries(value)) {
		const normalizedKey = validatePolicyKey(key, workspacePath, `${context}.${JSON.stringify(key)}`);
		entries.set(normalizedKey, normalizePolicyEntry(entry, `${context}.${JSON.stringify(key)}`));
	}
	return entries;
}

function readJsonFile(path: string): unknown | undefined {
	if (!existsSync(path)) return undefined;
	let raw: string;
	try {
		raw = readFileSync(path, "utf8");
	} catch (error) {
		throw new PolicyError(`Failed to read global P2P config ${path}`, { cause: error });
	}
	try {
		return JSON.parse(raw) as unknown;
	} catch (error) {
		throw new PolicyError(`Failed to parse global P2P config ${path}`, { cause: error });
	}
}

function readTomlFile(path: string): unknown | undefined {
	if (!existsSync(path)) return undefined;
	let raw: string;
	try {
		raw = readFileSync(path, "utf8");
	} catch (error) {
		throw new PolicyError(`Failed to read P2P policy ${path}`, { cause: error });
	}
	try {
		return parseToml(raw) as unknown;
	} catch (error) {
		throw new PolicyError(`Failed to parse P2P policy TOML ${path}`, { cause: error });
	}
}

function readPolicyEntries(parsed: unknown, workspacePath: string, path: string): Map<string, PolicyEntry> {
	if (parsed === undefined) return new Map();
	if (!isRecord(parsed)) throw new PolicyError(`P2P policy ${path} must contain an object`);
	const policySection = parsed.policy ?? (isRecord(parsed.p2p) ? parsed.p2p.policy : undefined);
	if (policySection !== undefined) return normalizePolicyEntries(policySection, workspacePath, `${path} [policy]`);

	const entries = new Map<string, PolicyEntry>();
	for (const [key, value] of Object.entries(parsed)) {
		if (key === "quotas" || key === "newFilesPublic" || key === "p2p") continue;
		const normalizedKey = validatePolicyKey(key, workspacePath, `${path} ${JSON.stringify(key)}`);
		entries.set(normalizedKey, normalizePolicyEntry(value, `${path} ${JSON.stringify(key)}`));
	}
	return entries;
}

function readGlobalPolicy(
	parsed: unknown,
	workspacePath: string,
	path: string,
): { entries: Map<string, PolicyEntry>; p2p: Record<string, unknown> } {
	if (parsed === undefined) return { entries: new Map(), p2p: {} };
	if (!isRecord(parsed)) throw new PolicyError(`Global config ${path} must contain an object`);
	const p2p = parsed.p2p;
	if (p2p === undefined) return { entries: new Map(), p2p: {} };
	if (!isRecord(p2p)) throw new PolicyError(`Global config ${path}.p2p must be an object`);
	return {
		entries: normalizePolicyEntries(p2p.policy ?? {}, workspacePath, `${path} p2p.policy`),
		p2p,
	};
}

function readWorkspaceNewFilesPublic(parsed: unknown, path: string): boolean | undefined {
	if (!isRecord(parsed)) return undefined;
	const p2p = isRecord(parsed.p2p) ? parsed.p2p : undefined;
	const configured = parsed.newFilesPublic ?? p2p?.newFilesPublic;
	if (configured === undefined) return undefined;
	if (typeof configured !== "boolean") throw new PolicyError(`P2P policy ${path} newFilesPublic must be a boolean`);
	return configured;
}

function readWorkspaceQuotas(parsed: unknown, path: string): Record<string, unknown> {
	if (!isRecord(parsed)) return {};
	const quotas = parsed.quotas ?? (isRecord(parsed.p2p) ? parsed.p2p.quotas : undefined);
	if (quotas !== undefined) {
		if (!isRecord(quotas)) throw new PolicyError(`P2P policy ${path} quotas must be an object`);
		return quotas;
	}
	const directKeys = [
		"queriesPerHour",
		"queries_per_hour",
		"burst",
		"maxBodyBytes",
		"max_body_bytes",
		"maxFileBytes",
		"max_file_bytes",
	];
	return Object.fromEntries(directKeys.filter((key) => parsed[key] !== undefined).map((key) => [key, parsed[key]]));
}

function mergeQuotas(
	globalP2p: Record<string, unknown>,
	workspace: Record<string, unknown>,
	path: string,
): PolicyQuotas {
	const globalQuotas = globalP2p.quotas;
	if (globalQuotas !== undefined && !isRecord(globalQuotas)) {
		throw new PolicyError("Global config p2p.quotas must be an object");
	}
	const globalValues: Record<string, unknown> = {
		...(isRecord(globalQuotas) ? globalQuotas : {}),
		...(globalP2p.maxBodyBytes !== undefined ? { maxBodyBytes: globalP2p.maxBodyBytes } : {}),
		...(globalP2p.maxFileBytes !== undefined ? { maxFileBytes: globalP2p.maxFileBytes } : {}),
	};
	const aliases: Record<keyof PolicyQuotas, readonly string[]> = {
		queriesPerHour: ["queriesPerHour", "queries_per_hour"],
		burst: ["burst"],
		maxBodyBytes: ["maxBodyBytes", "max_body_bytes"],
		maxFileBytes: ["maxFileBytes", "max_file_bytes"],
	};
	const knownKeys = new Set(Object.values(aliases).flat());
	for (const key of Object.keys(globalValues)) {
		if (!knownKeys.has(key))
			throw new PolicyError(`Global config p2p.quotas contains unknown field ${JSON.stringify(key)}`);
	}
	for (const key of Object.keys(workspace)) {
		if (!knownKeys.has(key)) throw new PolicyError(`${path} quotas contains unknown field ${JSON.stringify(key)}`);
	}
	const result = {} as Record<keyof PolicyQuotas, number>;
	for (const key of Object.keys(DEFAULT_QUOTAS) as Array<keyof PolicyQuotas>) {
		const workspaceKey = aliases[key].find((candidate) => workspace[candidate] !== undefined);
		const globalKey = aliases[key].find((candidate) => globalValues[candidate] !== undefined);
		const value =
			workspaceKey === undefined
				? globalKey === undefined
					? DEFAULT_QUOTAS[key]
					: globalValues[globalKey]
				: workspace[workspaceKey];
		if (!isPositiveInteger(value)) {
			throw new PolicyError(`${path}.${key} must be a positive safe integer`);
		}
		result[key] = value;
	}
	return result;
}

function globToRegExp(pattern: string): RegExp {
	let expression = "^";
	for (let index = 0; index < pattern.length; index += 1) {
		const character = pattern[index];
		if (character === "*" && pattern[index + 1] === "*") {
			index += 1;
			if (pattern[index + 1] === "/") {
				expression += "(?:.*/)?";
				index += 1;
			} else {
				expression += ".*";
			}
			continue;
		}
		if (character === "*") {
			expression += "[^/]*";
			continue;
		}
		if (character === "?") {
			expression += "[^/]";
			continue;
		}
		expression += /[\\^$+?.()|{}[\]]/u.test(character) ? `\\${character}` : character;
	}
	return new RegExp(`${expression}$`, "u");
}

function loadSeenSources(path: string, workspacePath: string): SeenSourceState {
	if (!existsSync(path)) return { promoted: new Set(), pending: new Set() };
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
	} catch (error) {
		throw new PolicyError(`Failed to parse seen P2P sources ${path}`, { cause: error });
	}
	const sourceList = isRecord(parsed) ? parsed.sources : parsed;
	if (!Array.isArray(sourceList)) throw new PolicyError(`Seen P2P sources ${path} must be an array`);
	const pendingList = isRecord(parsed) ? parsed.pending : undefined;
	if (pendingList !== undefined && !Array.isArray(pendingList)) {
		throw new PolicyError(`Seen P2P sources ${path} pending must be an array`);
	}
	const promoted = new Set<string>();
	for (const [index, source] of sourceList.entries()) {
		promoted.add(validatePolicyKey(source, workspacePath, `${path} sources[${index}]`));
	}
	const pending = new Set<string>();
	for (const [index, source] of (pendingList ?? []).entries()) {
		const normalized = validatePolicyKey(source, workspacePath, `${path} pending[${index}]`);
		if (!promoted.has(normalized)) pending.add(normalized);
	}
	return { promoted, pending };
}

function resolveGlobalConfigPath(homePath: string | undefined): string {
	if (homePath === undefined) return join(resolveAutoRAGHome(), "config.json");
	return basename(homePath) === ".autorag" ? join(homePath, "config.json") : join(homePath, ".autorag", "config.json");
}

export class PolicyStore {
	readonly quotas: PolicyQuotas;
	readonly newFilesPublic: boolean;
	private readonly entries: Map<string, PolicyEntry>;
	private readonly seenSources: Set<string>;
	private readonly pendingSources: Set<string>;
	private readonly seenSourcesPath: string;
	private readonly workspacePath: string;
	private readonly compiledPatterns: readonly {
		readonly pattern: string;
		readonly entry: PolicyEntry;
		readonly matcher: RegExp;
		readonly order: number;
	}[];

	constructor(options: PolicyStoreOptions | string, homePath?: string) {
		const normalizedOptions: PolicyStoreOptions =
			typeof options === "string" ? { workspacePath: options, homePath } : options;
		if (typeof normalizedOptions.workspacePath !== "string" || normalizedOptions.workspacePath.length === 0) {
			throw new PolicyError("workspacePath must be a non-empty path");
		}
		this.workspacePath = resolve(normalizedOptions.workspacePath);
		const globalConfigPath =
			normalizedOptions.globalConfigPath ?? resolveGlobalConfigPath(normalizedOptions.homePath);
		const workspacePolicyPath =
			normalizedOptions.workspacePolicyPath ?? join(this.workspacePath, ".autorag", "p2p", "policy.toml");
		const global = readGlobalPolicy(readJsonFile(globalConfigPath), this.workspacePath, globalConfigPath);
		const workspacePolicy = readTomlFile(workspacePolicyPath);
		const workspaceEntries = readPolicyEntries(workspacePolicy, this.workspacePath, workspacePolicyPath);
		this.entries = new Map(global.entries);
		for (const [pattern, entry] of workspaceEntries) this.entries.set(pattern, entry);
		this.quotas = mergeQuotas(
			global.p2p,
			readWorkspaceQuotas(workspacePolicy, workspacePolicyPath),
			workspacePolicyPath,
		);
		const configuredNewFilesPublic = global.p2p.newFilesPublic;
		if (configuredNewFilesPublic !== undefined && typeof configuredNewFilesPublic !== "boolean") {
			throw new PolicyError("Global config p2p.newFilesPublic must be a boolean");
		}
		const workspaceNewFilesPublic = readWorkspaceNewFilesPublic(workspacePolicy, workspacePolicyPath);
		if (normalizedOptions.newFilesPublic !== undefined && typeof normalizedOptions.newFilesPublic !== "boolean") {
			throw new PolicyError("newFilesPublic must be a boolean");
		}
		this.newFilesPublic =
			normalizedOptions.newFilesPublic ?? workspaceNewFilesPublic ?? configuredNewFilesPublic ?? false;
		this.seenSourcesPath =
			normalizedOptions.seenSourcesPath ?? join(this.workspacePath, ".autorag", "p2p", "seen-sources.json");
		const seenState = loadSeenSources(this.seenSourcesPath, this.workspacePath);
		this.seenSources = seenState.promoted;
		this.pendingSources = seenState.pending;
		this.compiledPatterns = [...this.entries.entries()].map(([pattern, entry], order) => {
			try {
				return { pattern, entry, matcher: globToRegExp(pattern), order };
			} catch (error) {
				throw new PolicyError(`Invalid source-identifier glob ${JSON.stringify(pattern)}`, { cause: error });
			}
		});
	}

	/** Record an indexer's first observation; allow-globs remain private until promotion. */
	markSourceSeen(source: string): void {
		const normalized = validatePolicyKey(source, this.workspacePath, "source");
		if (this.seenSources.has(normalized) || this.pendingSources.has(normalized)) return;
		this.pendingSources.add(normalized);
		try {
			this.persistSeenSources();
		} catch (error) {
			this.pendingSources.delete(normalized);
			throw error;
		}
	}

	private persistSeenSources(): void {
		try {
			mkdirSync(dirname(this.seenSourcesPath), { recursive: true });
			writeFileSync(
				this.seenSourcesPath,
				`${JSON.stringify(
					{
						sources: [...this.seenSources].sort(),
						pending: [...this.pendingSources].sort(),
					},
					null,
					2,
				)}\n`,
				{
					mode: 0o600,
				},
			);
		} catch (error) {
			throw new PolicyError("Failed to persist seen P2P sources", { cause: error });
		}
	}

	/** Alias used by indexers that report observations rather than seen-state changes. */
	observeSource(source: string): void {
		this.markSourceSeen(source);
	}

	/** Short alias for indexer integrations. */
	markSeen(source: string): void {
		this.markSourceSeen(source);
	}

	/** Explicitly promote a newly indexed source into normal allow-glob policy evaluation. */
	promoteSource(source: string): void {
		const normalized = validatePolicyKey(source, this.workspacePath, "source");
		if (this.seenSources.has(normalized)) return;
		this.pendingSources.delete(normalized);
		this.seenSources.add(normalized);
		try {
			this.persistSeenSources();
		} catch (error) {
			this.seenSources.delete(normalized);
			this.pendingSources.add(normalized);
			throw error;
		}
	}

	isSourceSeen(source: string): boolean {
		const normalized = validatePolicyKey(source, this.workspacePath, "source");
		return this.seenSources.has(normalized) || this.pendingSources.has(normalized);
	}

	getQuotas(): PolicyQuotas {
		return { ...this.quotas };
	}

	getEffectivePolicy(): Readonly<Record<string, PolicyEntry>> {
		return Object.fromEntries(this.entries);
	}

	getPolicy(): Readonly<Record<string, PolicyEntry>> {
		return this.getEffectivePolicy();
	}

	resolvePolicy(source: string, peerFingerprint?: string): PolicyResolution {
		if (typeof source !== "string" || source.length === 0 || UNSAFE_POLICY_CHARACTERS.test(source)) {
			return privateResolution();
		}
		const matches: PolicyMatch[] = this.compiledPatterns
			.filter((candidate) => candidate.matcher.test(source))
			.map((candidate) => ({ pattern: candidate.pattern, entry: candidate.entry, order: candidate.order }))
			.sort((left, right) => {
				const lengthDifference = right.pattern.length - left.pattern.length;
				if (lengthDifference !== 0) return lengthDifference;
				if (left.entry.tier === "never" && right.entry.tier !== "never") return -1;
				if (right.entry.tier === "never" && left.entry.tier !== "never") return 1;
				return left.order - right.order;
			});
		const selected = matches[0]?.entry;
		if (selected === undefined) return privateResolution();
		const isNewAllowSource = selected.tier === "always" || selected.tier === "peers";
		if (isNewAllowSource && !this.newFilesPublic && !this.seenSources.has(source.normalize("NFC"))) {
			return privateResolution();
		}
		return resolveEntry(selected, peerFingerprint);
	}
}

function privateResolution(): PolicyResolution {
	return { tier: "private", allowed: false, shareBytes: false, redact: true };
}

function resolveEntry(entry: PolicyEntry, peerFingerprint: string | undefined): PolicyResolution {
	switch (entry.tier) {
		case "private":
		case "never":
			return { tier: entry.tier, allowed: false, shareBytes: false, redact: true };
		case "always":
			return { tier: "always", allowed: true, shareBytes: true, redact: false };
		case "peers":
			return {
				tier: "peers",
				allowed: peerFingerprint !== undefined && (entry.peers ?? []).includes(peerFingerprint),
				shareBytes: false,
				redact: true,
			};
	}
}

export function createPolicyStore(options: PolicyStoreOptions | string, homePath?: string): PolicyStore {
	return new PolicyStore(options, homePath);
}

export function loadPolicyStore(options: PolicyStoreOptions | string, homePath?: string): PolicyStore {
	return new PolicyStore(options, homePath);
}

export const DEFAULT_POLICY_QUOTAS: PolicyQuotas = Object.freeze({ ...DEFAULT_QUOTAS });
