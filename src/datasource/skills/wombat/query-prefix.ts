import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parse, stringify } from "smol-toml";
import { resolveEmbeddingPrefixes } from "../../../embedding/prefixes.ts";

/**
 * wombat's MinSync workspace is created by `minsync init`, which writes no query prefix for Qwen3-Embedding. The
 * model is instruction-aware, so a bare query embeds worse than `Instruct: …\nQuery:<query>` (the same instruction
 * AutoRAG gives every other Qwen3 path; see `embedding/prefixes.ts`). MinSync prepends `query_prefix` to queries
 * only, so setting it re-embeds nothing: the passage vectors and the manifest hash are untouched.
 *
 * Only a default local Qwen3 workspace with no `query_prefix` is changed. An operator's value, including `""`, wins,
 * and any other model, a remote `base_url`, or a sparse-only workspace is left alone. Returns whether the file changed.
 */
export function applyWombatQueryPrefix(mirrorDir: string): boolean {
	const path = join(mirrorDir, ".minsync", "config.toml");
	if (!existsSync(path)) return false;
	try {
		const doc = parse(readFileSync(path, "utf8")) as Record<string, unknown>;
		const embedder = doc.embedder;
		if (typeof embedder !== "object" || embedder === null || Array.isArray(embedder)) return false;
		const section = embedder as Record<string, unknown>;
		if (typeof section.id !== "string" || Object.hasOwn(section, "query_prefix")) return false;
		const baseUrl = typeof section.base_url === "string" ? section.base_url : undefined;
		const prefix = resolveEmbeddingPrefixes({ id: section.id, ...(baseUrl === undefined ? {} : { baseUrl }) }).query;
		if (prefix === "") return false;
		section.query_prefix = prefix;
		writeFileSync(path, stringify(doc));
		return true;
	} catch {
		return false;
	}
}
