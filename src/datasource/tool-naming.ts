/**
 * Canonical model-visible search tool name for one datasource connection.
 *
 * The agent generates one `search_datasource_<id>` tool per authorized
 * connection (see `src/agent/search-single-datasource-tool.ts`), and skill
 * manifests must be able to name that tool without importing the agent layer
 * (the import direction is agent → datasource, never the reverse). The alias
 * rewrite in `aliased-skill.ts` relies on this exact transformation, so keep
 * it stable: lowercase, non-alphanumerics collapse to `_`, edges trimmed.
 */
export function datasourceSearchToolName(datasourceId: string): string {
	const collapsed = datasourceId.toLowerCase().replace(/[^a-z0-9]+/g, "_");
	// Trim edge underscores with explicit loops instead of an anchored
	// alternation: `/^_+|_+$/g` trips polynomial-ReDoS static analysis
	// (CodeQL js/polynomial-redos) on `_`-heavy input.
	let start = 0;
	let end = collapsed.length;
	while (start < end && collapsed[start] === "_") start += 1;
	while (end > start && collapsed[end - 1] === "_") end -= 1;
	const sanitized = collapsed.slice(start, end);
	return `search_datasource_${sanitized}`;
}
