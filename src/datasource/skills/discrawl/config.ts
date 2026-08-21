import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { parse, stringify } from "smol-toml";
import type { DiscrawlOptions } from "./types.ts";
import { DEFAULT_DISCRAWL_EMBEDDING_MODEL, DEFAULT_DISCRAWL_EMBEDDING_PROVIDER } from "./types.ts";

type TomlRecord = Record<string, unknown>;

/**
 * Creates or updates AutoRAG's workspace-local discrawl config.
 *
 * Only the fields AutoRAG owns are changed: the workspace database path and
 * embedding enablement/provider/model. Existing operator settings such as
 * guilds, source filters, sync policy, and search mode are preserved.
 */
export function ensureManagedDiscrawlConfig(configPath: string, databasePath: string, options: DiscrawlOptions): void {
	let parsed: TomlRecord = {};
	try {
		parsed = parse(readFileSync(configPath, "utf8")) as TomlRecord;
	} catch (error) {
		if (!isMissingFile(error)) throw error;
	}

	if (parsed.db_path === undefined) parsed.db_path = databasePath;
	const search = objectSection(parsed, "search");
	const embeddings = objectSection(search, "embeddings");
	embeddings.enabled = true;
	embeddings.provider = options.embeddingProvider ?? DEFAULT_DISCRAWL_EMBEDDING_PROVIDER;
	embeddings.model = options.embeddingModel ?? DEFAULT_DISCRAWL_EMBEDDING_MODEL;

	mkdirSync(dirname(configPath), { recursive: true });
	const temporaryPath = join(dirname(configPath), `.${randomUUID()}.tmp`);
	try {
		writeFileSync(temporaryPath, stringify(parsed), { encoding: "utf8", mode: 0o600 });
		renameSync(temporaryPath, configPath);
	} finally {
		try {
			unlinkSync(temporaryPath);
		} catch {}
	}
}

function objectSection(parent: TomlRecord, key: string): TomlRecord {
	const existing = parent[key];
	if (typeof existing === "object" && existing !== null && !Array.isArray(existing)) {
		return existing as TomlRecord;
	}
	const section: TomlRecord = {};
	parent[key] = section;
	return section;
}

function isMissingFile(error: unknown): boolean {
	return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}
