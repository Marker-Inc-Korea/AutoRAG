import { existsSync, readFileSync, realpathSync } from "node:fs";
import { basename, dirname, isAbsolute, join, matchesGlob, relative, resolve } from "node:path";
import { resolveAutoRAGHome, resolveUserHome } from "../config/home.ts";
import { writeFileAtomic } from "../mirror/atomic-write.ts";

export type AutomationEventType = "create" | "update" | "delete";

export const AUTOMATION_EVENT_TYPES: readonly AutomationEventType[] = ["create", "update", "delete"];

export interface AutomationEvent {
	readonly type: AutomationEventType;
	readonly path: string;
}

export interface Automation {
	readonly id: string;
	readonly paths: readonly string[];
	readonly events: readonly AutomationEventType[];
	readonly include: readonly string[];
	readonly exclude: readonly string[];
	readonly run: readonly string[];
	readonly appendFiles: boolean;
	readonly debounceMs: number;
	readonly timeoutMs: number;
}

export const DEFAULT_DEBOUNCE_MS = 1000;
export const DEFAULT_TIMEOUT_MS = 600_000;

export const DEFAULT_EXCLUDES: readonly string[] = [
	"**/.git/**",
	"**/node_modules/**",
	"**/.autorag/**",
	"**/*.crdownload",
	"**/*.part",
	"**/*.tmp",
	"**/~$*",
	"**/.DS_Store",
];

const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const KNOWN_KEYS: Readonly<Record<string, true>> = {
	id: true,
	paths: true,
	events: true,
	include: true,
	exclude: true,
	run: true,
	appendFiles: true,
	debounceMs: true,
	timeoutMs: true,
};

export class AutomationConfigError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "AutomationConfigError";
	}
}

export function automationsConfigPath(env: NodeJS.ProcessEnv = process.env): string {
	return join(resolveAutoRAGHome(env), "automations.json");
}

function expandTilde(value: string, home: string): string {
	if (value === "~") return home;
	if (value.startsWith("~/") || value.startsWith("~\\")) return join(home, value.slice(2));
	return value;
}

function requireStringArray(value: unknown, field: string, nonEmpty: boolean): string[] {
	if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.trim().length === 0)) {
		throw new AutomationConfigError(`Automation "${field}" must be an array of non-empty strings`);
	}
	if (nonEmpty && value.length === 0) {
		throw new AutomationConfigError(`Automation "${field}" must not be empty`);
	}
	return value as string[];
}

function parsePositiveInt(value: unknown, field: string, fallback: number): number {
	if (value === undefined) return fallback;
	if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
		throw new AutomationConfigError(`Automation "${field}" must be a positive integer`);
	}
	return value;
}

export function parseAutomations(raw: unknown, env: NodeJS.ProcessEnv = process.env): Automation[] {
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new AutomationConfigError("automations.json must be a JSON object with an `automations` array");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (key !== "automations") {
			throw new AutomationConfigError(`Unknown top-level field in automations.json: "${key}"`);
		}
	}
	if (!Array.isArray(record.automations)) {
		throw new AutomationConfigError("`automations` must be an array");
	}

	const userHome = resolveUserHome(env);
	const seenIds = new Set<string>();
	return record.automations.map((entry, idx) => {
		if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
			throw new AutomationConfigError(`automations[${idx}] must be an object`);
		}
		const obj = entry as Record<string, unknown>;
		for (const key of Object.keys(obj)) {
			if (!KNOWN_KEYS[key]) {
				throw new AutomationConfigError(`Unknown field "${key}" in automations[${idx}]`);
			}
		}
		if (typeof obj.id !== "string" || !ID_PATTERN.test(obj.id)) {
			throw new AutomationConfigError(
				`automations[${idx}].id must match ${ID_PATTERN} (got ${JSON.stringify(obj.id)})`,
			);
		}
		if (seenIds.has(obj.id)) {
			throw new AutomationConfigError(`Duplicate automation id: "${obj.id}"`);
		}
		seenIds.add(obj.id);

		const rawPaths = requireStringArray(obj.paths, `${obj.id}.paths`, true);
		const paths = rawPaths.map((p) => {
			const expanded = expandTilde(p, userHome);
			if (!isAbsolute(expanded)) {
				throw new AutomationConfigError(`Automation "${obj.id}" watch path must be absolute or ~/…: "${p}"`);
			}
			return resolve(expanded);
		});

		const eventsRaw =
			obj.events === undefined
				? [...AUTOMATION_EVENT_TYPES]
				: requireStringArray(obj.events, `${obj.id}.events`, true);
		for (const ev of eventsRaw) {
			if (!(AUTOMATION_EVENT_TYPES as readonly string[]).includes(ev)) {
				throw new AutomationConfigError(`Automation "${obj.id}" events must be create|update|delete (got "${ev}")`);
			}
		}

		const include = obj.include === undefined ? ["**"] : requireStringArray(obj.include, `${obj.id}.include`, true);
		const exclude = obj.exclude === undefined ? [] : requireStringArray(obj.exclude, `${obj.id}.exclude`, false);
		const run = requireStringArray(obj.run, `${obj.id}.run`, true).map((token) => expandTilde(token, userHome));

		if (obj.appendFiles !== undefined && typeof obj.appendFiles !== "boolean") {
			throw new AutomationConfigError(`Automation "${obj.id}.appendFiles" must be a boolean`);
		}

		return {
			id: obj.id,
			paths,
			events: eventsRaw as AutomationEventType[],
			include,
			exclude,
			run,
			appendFiles: obj.appendFiles === true,
			debounceMs: parsePositiveInt(obj.debounceMs, `${obj.id}.debounceMs`, DEFAULT_DEBOUNCE_MS),
			timeoutMs: parsePositiveInt(obj.timeoutMs, `${obj.id}.timeoutMs`, DEFAULT_TIMEOUT_MS),
		};
	});
}

function readRawAutomations(configPath: string): Record<string, unknown>[] {
	if (!existsSync(configPath)) return [];
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(configPath, "utf8"));
	} catch (error) {
		const msg = error instanceof Error ? error.message : String(error);
		throw new AutomationConfigError(`Invalid JSON in ${configPath}: ${msg}`);
	}
	if (
		typeof parsed !== "object" ||
		parsed === null ||
		!("automations" in parsed) ||
		!Array.isArray(parsed.automations)
	) {
		throw new AutomationConfigError(`${configPath} must be a JSON object with an \`automations\` array`);
	}
	return parsed.automations;
}

export function loadAutomations(
	configPath: string = automationsConfigPath(),
	env: NodeJS.ProcessEnv = process.env,
): Automation[] {
	return parseAutomations({ automations: readRawAutomations(configPath) }, env);
}

/**
 * Insert or replace the rule with `id`, keeping every other rule as written. The
 * whole file is validated before it is written, so an invalid rule never lands
 * on disk and the running daemon never reloads a broken file.
 */
export function saveAutomation(
	configPath: string,
	id: string,
	rule: Readonly<Record<string, unknown>>,
	env: NodeJS.ProcessEnv = process.env,
): Automation {
	const entry = { ...rule, id };
	const raw = readRawAutomations(configPath);
	const index = raw.findIndex((item) => item.id === id);
	const next = index === -1 ? [...raw, entry] : raw.map((item, i) => (i === index ? entry : item));
	const parsed = parseAutomations({ automations: next }, env);
	writeFileAtomic(configPath, `${JSON.stringify({ automations: next }, null, 2)}\n`);
	return parsed.find((item) => item.id === id) as Automation;
}

export function deleteAutomation(configPath: string, id: string): void {
	const raw = readRawAutomations(configPath);
	const next = raw.filter((item) => item.id !== id);
	if (next.length === raw.length) throw new AutomationConfigError(`Unknown automation id: "${id}"`);
	writeFileAtomic(configPath, `${JSON.stringify({ automations: next }, null, 2)}\n`);
}

/**
 * Match a single pattern against a POSIX-slash path relative to `watchedRoot`.
 * `matchesGlob("a.pdf", "**\/*.pdf")` is true on POSIX and false on Windows in
 * Node 24+, so try both the pattern and the stripped `**\/` form at root level.
 */
function globMatches(relPosix: string, pattern: string): boolean {
	if (pattern === "**") return true;
	if (matchesGlob(relPosix, pattern)) return true;
	if (pattern.startsWith("**/") && matchesGlob(relPosix, pattern.slice(3))) return true;
	return false;
}

function canonicalizePath(target: string): string {
	const resolved = resolve(target);
	try {
		return realpathSync(resolved);
	} catch {
		const parent = dirname(resolved);
		if (parent === resolved) return resolved;
		return join(canonicalizePath(parent), basename(resolved));
	}
}

export function matchesAutomationPath(automation: Automation, watchedRoot: string, filePath: string): boolean {
	const rel = relative(canonicalizePath(watchedRoot), canonicalizePath(filePath));
	if (rel.length === 0 || rel === "." || rel.startsWith("..") || isAbsolute(rel)) return false;
	const relPosix = rel.replaceAll("\\", "/");

	for (const pattern of DEFAULT_EXCLUDES) {
		if (globMatches(relPosix, pattern)) return false;
	}
	for (const pattern of automation.exclude) {
		if (globMatches(relPosix, pattern)) return false;
	}
	return automation.include.some((pattern) => globMatches(relPosix, pattern));
}

export function matchesAutomation(automation: Automation, watchedRoot: string, event: AutomationEvent): boolean {
	if (!automation.events.includes(event.type)) return false;
	return matchesAutomationPath(automation, watchedRoot, event.path);
}
