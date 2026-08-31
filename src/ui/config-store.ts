/**
 * Trusted config mutations for the local datasource UI.
 *
 * Reads/writes the operator `config.json` on disk. Never persists secret
 * values — only env-var names, paths, and lists. Enabling a connection
 * recomputes `datasourceAccess` from catalog tags + `/${alias}/**` scopes.
 */

import { ConfigError, readRawConfigObject, writeConfigObject } from "../cli/config.ts";
import { BUILTIN_DATASOURCE_SKILL_NAMES, type DatasourceSkillConfig } from "../datasource/skills/factory.ts";
import { DATASOURCE_TYPE_CATALOG, getDatasourceType, SOURCE_PICKER } from "./catalog.ts";
import { type ProbeResult, probeConnection } from "./probe.ts";

const ALIAS_PATTERN = /^[a-z][a-z0-9-]{0,62}$/;
const SECRET_KEYS = new Set([
	"token",
	"password",
	"accesstoken",
	"apikey",
	"secret",
	"credential",
	"authorization",
	"cookie",
	"privatekey",
	"refreshtoken",
	"env",
]);
const BUILTIN = new Set<string>(BUILTIN_DATASOURCE_SKILL_NAMES);

export interface ConnectionInput {
	readonly alias: string;
	readonly type: string;
	readonly enabled?: boolean;
	readonly instanceId?: string;
	readonly connector?: Record<string, unknown>;
	readonly tags?: readonly string[];
}

export interface UiConnection {
	readonly alias: string;
	readonly type: string;
	readonly enabled: boolean;
	readonly instanceId?: string;
	readonly connector: Record<string, unknown>;
	readonly tags: readonly string[];
	readonly probe: ProbeResult;
}

export interface UiState {
	readonly searchPaths: readonly string[];
	readonly connections: readonly UiConnection[];
	readonly catalog: typeof DATASOURCE_TYPE_CATALOG;
	readonly picker: typeof SOURCE_PICKER;
	readonly access: { readonly allowedTags: readonly string[]; readonly allowedScopes: readonly string[] };
}

export function listUiState(configPath: string, env: NodeJS.ProcessEnv = process.env): UiState {
	const raw = readRawConfigObject(configPath);
	const searchPaths = asStringArray(raw.searchPaths);
	const connections = [...readConnections(raw)].map((connection) => ({
		...connection,
		probe: probeConnection(connection, { env }),
	}));
	const access = accessFromEnabled(connections.filter((item) => item.enabled));
	return { searchPaths, connections, catalog: DATASOURCE_TYPE_CATALOG, picker: SOURCE_PICKER, access };
}

export function upsertConnection(configPath: string, input: ConnectionInput): UiState {
	const alias = normalizeAlias(input.alias);
	const type = input.type.trim();
	if (!BUILTIN.has(type)) throw new ConfigError(`Unknown datasource type: ${type}`);
	const raw = readRawConfigObject(configPath);
	const datasources = asObject(raw.datasources);
	const connector = normalizeConnector(type, stripSecrets(input.connector ?? {}));
	const enabled = input.enabled !== false;
	const entry: Record<string, unknown> = {
		type,
		enabled,
		connector,
	};
	if (typeof input.instanceId === "string" && input.instanceId.trim().length > 0) {
		entry.instanceId = input.instanceId.trim();
	}
	if (input.tags !== undefined && input.tags.length > 0) entry.tags = [...input.tags];
	datasources[alias] = entry;
	raw.datasources = datasources;
	writeAccess(
		raw,
		readConnections(raw).filter((item) => item.enabled),
	);
	writeConfigObject(configPath, raw);
	return listUiState(configPath);
}

export function toggleConnection(configPath: string, alias: string, enabled: boolean): UiState {
	const raw = readRawConfigObject(configPath);
	const datasources = asObject(raw.datasources);
	const current = datasources[alias];
	if (current === undefined) throw new ConfigError(`Unknown connection: ${alias}`);
	if (current === false || current === true) {
		datasources[alias] = { type: alias, enabled };
	} else if (typeof current === "object" && current !== null && !Array.isArray(current)) {
		datasources[alias] = { ...current, enabled };
	} else {
		throw new ConfigError(`Unknown connection: ${alias}`);
	}
	raw.datasources = datasources;
	writeAccess(
		raw,
		readConnections(raw).filter((item) => item.enabled),
	);
	writeConfigObject(configPath, raw);
	return listUiState(configPath);
}

export function removeConnection(configPath: string, alias: string): UiState {
	const raw = readRawConfigObject(configPath);
	const datasources = asObject(raw.datasources);
	if (datasources[alias] === undefined) throw new ConfigError(`Unknown connection: ${alias}`);
	delete datasources[alias];
	if (Object.keys(datasources).length === 0) delete raw.datasources;
	else raw.datasources = datasources;
	writeAccess(
		raw,
		readConnections(raw).filter((item) => item.enabled),
	);
	writeConfigObject(configPath, raw);
	return listUiState(configPath);
}

export function setSearchPaths(configPath: string, searchPaths: readonly string[]): UiState {
	const raw = readRawConfigObject(configPath);
	raw.searchPaths = searchPaths.map((item) => item.trim()).filter((item) => item.length > 0);
	writeConfigObject(configPath, raw);
	return listUiState(configPath);
}

function writeAccess(raw: Record<string, unknown>, enabled: readonly Omit<UiConnection, "probe">[]): void {
	const access = accessFromEnabled(enabled);
	if (access.allowedTags.length === 0) delete raw.datasourceAccess;
	else raw.datasourceAccess = { allowedTags: [...access.allowedTags], allowedScopes: [...access.allowedScopes] };
}

function accessFromEnabled(enabled: readonly { alias: string; type: string; tags: readonly string[] }[]): {
	allowedTags: string[];
	allowedScopes: string[];
} {
	const tags: string[] = [];
	const scopes: string[] = [];
	for (const item of enabled) {
		for (const tag of item.tags) {
			if (!tags.includes(tag)) tags.push(tag);
		}
		const scope = `/${item.alias}/**`;
		if (!scopes.includes(scope)) scopes.push(scope);
	}
	return { allowedTags: tags, allowedScopes: scopes };
}

function readConnections(raw: Record<string, unknown>): Omit<UiConnection, "probe">[] {
	const datasources = asObject(raw.datasources);
	const connections: Omit<UiConnection, "probe">[] = [];
	for (const [alias, value] of Object.entries(datasources)) {
		if (value === false) {
			connections.push({ alias, type: alias, enabled: false, connector: {}, tags: tagsFor(alias, {}) });
			continue;
		}
		if (value === true) {
			connections.push({ alias, type: alias, enabled: true, connector: {}, tags: tagsFor(alias, {}) });
			continue;
		}
		if (typeof value !== "object" || value === null || Array.isArray(value)) continue;
		const entry = value as DatasourceSkillConfig & { enabled?: boolean; type?: string };
		const type = typeof entry.type === "string" && entry.type.length > 0 ? entry.type : alias;
		const connector = stripSecrets(asObject(entry.connector as Record<string, unknown> | undefined));
		const enabled = entry.enabled !== false;
		connections.push({
			alias,
			type,
			enabled,
			...(typeof entry.instanceId === "string" ? { instanceId: entry.instanceId } : {}),
			connector,
			tags: Array.isArray(entry.tags) ? entry.tags.map(String) : tagsFor(type, connector),
		});
	}
	return connections;
}

function tagsFor(type: string, connector: Record<string, unknown>): string[] {
	const catalog = getDatasourceType(type);
	const tags = [...(catalog?.defaultTags ?? [type])];
	if (type === "cloud-drive" && typeof connector.provider === "string" && connector.provider.length > 0) {
		if (!tags.includes(connector.provider)) tags.push(connector.provider);
	}
	return tags;
}

export function normalizeAlias(alias: string): string {
	const trimmed = alias.trim();
	if (!ALIAS_PATTERN.test(trimmed)) {
		throw new ConfigError("Connection name must be lowercase letters, digits, and hyphens (start with a letter).");
	}
	return trimmed;
}

function normalizeConnector(type: string, connector: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = { ...connector };
	if (out.repos !== undefined) out.repos = asLineList(out.repos);
	if (out.paths !== undefined) out.paths = asLineList(out.paths);
	if (out.queries !== undefined) out.queries = asLineList(out.queries);
	if (out.labelIds !== undefined) out.labelIds = asLineList(out.labelIds);
	if (out.feeds !== undefined) out.feeds = asFeeds(out.feeds);
	if (out.backend === "") delete out.backend;
	if (out.includeSharedDrives === "true") out.includeSharedDrives = true;
	if (out.includeSharedDrives === "false") out.includeSharedDrives = false;
	for (const [key, value] of Object.entries(out)) {
		if (value === "" || value === undefined) delete out[key];
	}
	if (type === "github" && !Array.isArray(out.repos)) out.repos = [];
	return out;
}

function asFeeds(value: unknown): Array<{ url: string; category?: string }> {
	const lines: unknown[] = Array.isArray(value) ? value : typeof value === "string" ? value.split(/\r?\n/) : [];
	const feeds: Array<{ url: string; category?: string }> = [];
	for (const item of lines) {
		if (typeof item === "string") {
			const line = item.trim();
			if (line.length === 0) continue;
			const split = line.split("|", 2);
			const url = split[0]?.trim() ?? "";
			const category = split[1]?.trim();
			if (url.length === 0) continue;
			feeds.push(category ? { url, category } : { url });
			continue;
		}
		if (
			item &&
			typeof item === "object" &&
			!Array.isArray(item) &&
			typeof (item as { url?: unknown }).url === "string"
		) {
			const url = (item as { url: string }).url.trim();
			if (url.length === 0) continue;
			const category = (item as { category?: unknown }).category;
			feeds.push(typeof category === "string" && category.length > 0 ? { url, category } : { url });
		}
	}
	return feeds;
}

function asLineList(value: unknown): string[] {
	if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter((item) => item.length > 0);
	if (typeof value === "string") {
		return value
			.split(/\r?\n/)
			.map((item) => item.trim())
			.filter((item) => item.length > 0);
	}
	return [];
}

export function stripSecrets<T>(value: T): T {
	if (Array.isArray(value)) return value.map((item) => stripSecrets(item)) as T;
	if (value && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
			if (isSecretKey(key)) continue;
			out[key] = stripSecrets(nested);
		}
		return out as T;
	}
	return value;
}

function isSecretKey(key: string): boolean {
	const normalized = key.replace(/[-_]/g, "").toLowerCase();
	if (normalized.length > 3 && normalized.endsWith("env")) return false;
	return SECRET_KEYS.has(normalized) || normalized.includes("token") || normalized.includes("secret");
}

function asObject(value: unknown): Record<string, unknown> {
	if (typeof value === "object" && value !== null && !Array.isArray(value))
		return { ...(value as Record<string, unknown>) };
	return {};
}

function asStringArray(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value.map((item) => String(item));
}
