import { join } from "node:path";

export const AUTORAG_DIRNAME = ".autorag";
export const DATASOURCES_DIRNAME = "datasources";
export const DISCRAWL_DATASOURCE_DIRNAME = "discrawl";

/** Logical datasource kind for source identifiers. */
export const DISCRAWL_SOURCE_KIND = "discord" as const;

/**
 * Root directory for discrawl-managed state under a workspace. This is where
 * the external `discrawl` CLI stores its SQLite archive, FTS index, and
 * message vectors.
 */
export function discrawlDatasourceRoot(workspaceRoot: string): string {
	return join(workspaceRoot, AUTORAG_DIRNAME, DATASOURCES_DIRNAME, DISCRAWL_DATASOURCE_DIRNAME);
}

/** Config file discrawl reads for embedding provider/model settings. */
export function discrawlConfigPath(workspaceRoot: string): string {
	return join(discrawlDatasourceRoot(workspaceRoot), "config.toml");
}

/** SQLite archive maintained by the discrawl CLI. */
export function discrawlDatabasePath(workspaceRoot: string): string {
	return join(discrawlDatasourceRoot(workspaceRoot), "discrawl.db");
}

/**
 * Builds the slash-hierarchical source identifier for a single archived
 * message. Discord message ids are stable snowflakes, so these sources stay
 * traceable across re-syncs.
 */
export function discrawlSourcePath(instanceId: string, messageId: string): string {
	return `/${DISCRAWL_SOURCE_KIND}/${instanceId}/chunks/${messageId}`;
}

/**
 * Parses a discrawl source identifier back into its components. Returns
 * `undefined` for malformed or non-discord sources so deny decisions stay
 * explicit rather than undefined-as-deny.
 */
export function parseDiscrawlSourcePath(
	source: string,
): { readonly instanceId: string; readonly messageId: string } | undefined {
	const match = /^\/discord\/([^/]+)\/chunks\/([^/]+)$/.exec(source);
	if (match === null) return undefined;
	const [, instanceId, messageId] = match;
	if (instanceId === undefined || messageId === undefined) return undefined;
	return { instanceId, messageId };
}
