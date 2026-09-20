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
	const sanitized = datasourceId
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "");
	return `search_datasource_${sanitized}`;
}
