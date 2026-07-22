import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parse, stringify } from "smol-toml";
import type { MinSyncEmbedderConfig } from "./types.ts";

/**
 * MinSync config.toml lives at `<workspace>/.minsync/config.toml`.
 * After `minsync init` runs, we rewrite allowlisted embedder fields so
 * the caller's configuration wins over init defaults.
 */
export const MINSYNC_CONFIG_DIR = ".minsync";
export const MINSYNC_CONFIG_FILE = "config.toml";

export function minSyncConfigPath(workspacePath: string): string {
	return join(workspacePath, MINSYNC_CONFIG_DIR, MINSYNC_CONFIG_FILE);
}

/**
 * Atomically rewrite allowlisted embedder fields in MinSync config.toml.
 * Reads the existing file, merges the allowlisted fields from `embedder`,
 * and writes the result back. Only fields present on `embedder` are applied;
 * absent fields are left untouched.
 *
 * Allowlisted [embedder] fields: id, base_url, query_prefix, passage_prefix,
 * batch_size, max_retries, max_concurrent, timeout_seconds (from timeoutMs / 1000 ceil).
 * Dimension is written under [vectorstore.options] as `dimension`.
 *
 * Returns true if the file was rewritten, false if the file does not exist.
 */
export function rewriteEmbedderConfig(workspacePath: string, embedder: MinSyncEmbedderConfig): boolean {
	const configPath = minSyncConfigPath(workspacePath);
	let raw: string;
	try {
		raw = readFileSync(configPath, "utf8");
	} catch {
		return false;
	}

	const parsed = parse(raw) as Record<string, Record<string, unknown>>;
	parsed.embedder ??= {};
	const embedderSection = parsed.embedder as Record<string, unknown>;
	parsed.vectorstore ??= {};
	const vectorstore = parsed.vectorstore as Record<string, unknown>;
	const options =
		typeof vectorstore.options === "object" && vectorstore.options !== null && !Array.isArray(vectorstore.options)
			? (vectorstore.options as Record<string, unknown>)
			: {};
	vectorstore.options = options;

	if (embedder.id !== undefined) embedderSection.id = embedder.id;
	if (embedder.baseUrl !== undefined) embedderSection.base_url = embedder.baseUrl;
	if (embedder.queryPrefix !== undefined) embedderSection.query_prefix = embedder.queryPrefix;
	if (embedder.passagePrefix !== undefined) embedderSection.passage_prefix = embedder.passagePrefix;
	if (embedder.batchSize !== undefined) embedderSection.batch_size = embedder.batchSize;
	if (embedder.maxRetries !== undefined) embedderSection.max_retries = embedder.maxRetries;
	if (embedder.maxConcurrent !== undefined) embedderSection.max_concurrent = embedder.maxConcurrent;
	if (embedder.timeoutMs !== undefined) {
		embedderSection.timeout_seconds = Math.ceil(embedder.timeoutMs / 1000);
	}
	if (embedder.dimension !== undefined) options.dimension = embedder.dimension;

	writeFileSync(configPath, stringify(parsed));
	return true;
}
