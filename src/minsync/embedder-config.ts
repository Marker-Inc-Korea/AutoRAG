import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parse, stringify } from "smol-toml";
import type { MinSyncLexicalLanguage } from "./lexical-language.ts";
import type { MinSyncEmbedderConfig } from "./types.ts";

export interface MinSyncEmbeddingIdentity {
	readonly provider: string;
	readonly model: string;
	readonly artifactRevision: string;
	readonly dimension: number;
	readonly queryPrefix: string;
	readonly passagePrefix: string;
	readonly runtimeBuild: string;
}

export const EMBEDDING_IDENTITY_FILE = "autorag-embedding-identity.json";

export const DEFAULT_MINSYNC_EMBEDDER_ID = "native:Qwen/Qwen3-Embedding-0.6B";
export const DEFAULT_MINSYNC_EMBEDDER_DIMENSION = 1024;
export const DEFAULT_MINSYNC_EMBEDDER_CONFIG: MinSyncEmbedderConfig = {
	id: DEFAULT_MINSYNC_EMBEDDER_ID,
	dimension: DEFAULT_MINSYNC_EMBEDDER_DIMENSION,
};

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

export function minSyncEmbeddingIdentityPath(workspacePath: string): string {
	return join(workspacePath, MINSYNC_CONFIG_DIR, EMBEDDING_IDENTITY_FILE);
}

export function configuredMaxChunkSize(workspacePath: string): number | undefined {
	let raw: string;
	try {
		raw = readFileSync(minSyncConfigPath(workspacePath), "utf8");
	} catch {
		return undefined;
	}
	const parsed = parse(raw) as Record<string, unknown>;
	const chunker = parsed.chunker;
	if (typeof chunker !== "object" || chunker === null || Array.isArray(chunker)) return undefined;
	const options = (chunker as Record<string, unknown>).options;
	if (typeof options !== "object" || options === null || Array.isArray(options)) return undefined;
	const maxChunkSize = (options as Record<string, unknown>).max_chunk_size;
	return typeof maxChunkSize === "number" ? maxChunkSize : undefined;
}

export function configuredVectorDimension(workspacePath: string): number | undefined {
	let raw: string;
	try {
		raw = readFileSync(minSyncConfigPath(workspacePath), "utf8");
	} catch {
		return undefined;
	}
	const parsed = parse(raw) as Record<string, unknown>;
	const vectorstore = parsed.vectorstore;
	if (typeof vectorstore !== "object" || vectorstore === null || Array.isArray(vectorstore)) return undefined;
	const options = (vectorstore as Record<string, unknown>).options;
	if (typeof options !== "object" || options === null || Array.isArray(options)) return undefined;
	const dimension = (options as Record<string, unknown>).dimension;
	return typeof dimension === "number" ? dimension : undefined;
}

/**
 * The BM25 tokenizer in `[lexical].language`, or undefined when there is no config. A config without the section
 * (written before MinSync had one) behaves as MinSync's default, `simple`.
 */
export function configuredLexicalLanguage(workspacePath: string): string | undefined {
	let raw: string;
	try {
		raw = readFileSync(minSyncConfigPath(workspacePath), "utf8");
	} catch {
		return undefined;
	}
	const lexical = (parse(raw) as Record<string, unknown>).lexical;
	if (typeof lexical !== "object" || lexical === null || Array.isArray(lexical)) return "simple";
	const language = (lexical as Record<string, unknown>).language;
	return typeof language === "string" ? language : "simple";
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
 * `managed.maxChunkSize` goes to [chunker.options] and `managed.lexicalLanguage` to [lexical].language. MinSync
 * itself records the tokenizer an index was built with in its cursor and, when [lexical].language differs, deletes
 * and re-indexes every chunk on the next sync, so changing it needs no `--full` from the caller.
 *
 * Returns true if the file was rewritten, false if the file does not exist.
 */
export function rewriteEmbedderConfig(
	workspacePath: string,
	embedder: MinSyncEmbedderConfig,
	managed: { readonly maxChunkSize?: number; readonly lexicalLanguage?: MinSyncLexicalLanguage } = {},
): boolean {
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
	const chunker =
		typeof parsed.chunker === "object" && parsed.chunker !== null && !Array.isArray(parsed.chunker)
			? (parsed.chunker as Record<string, unknown>)
			: {};
	parsed.chunker = chunker;
	const chunkerOptions =
		typeof chunker.options === "object" && chunker.options !== null && !Array.isArray(chunker.options)
			? (chunker.options as Record<string, unknown>)
			: {};
	chunker.options = chunkerOptions;

	if (embedder.id?.startsWith("native:")) {
		delete embedderSection.base_url;
		delete embedderSection.query_prefix;
		delete embedderSection.passage_prefix;
	}
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
	if (managed.maxChunkSize !== undefined) chunkerOptions.max_chunk_size = managed.maxChunkSize;
	if (managed.lexicalLanguage !== undefined) {
		const lexical =
			typeof parsed.lexical === "object" && parsed.lexical !== null && !Array.isArray(parsed.lexical)
				? (parsed.lexical as Record<string, unknown>)
				: {};
		parsed.lexical = lexical;
		lexical.language = managed.lexicalLanguage;
	}

	writeFileSync(configPath, stringify(parsed));
	return true;
}
