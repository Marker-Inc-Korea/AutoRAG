/**
 * Turns a datasource's `connector.sidecar` config into {@link CrawlerSidecarOptions}.
 *
 * On by default for every datasource that has a crawler descriptor; opt out with
 *
 *   "connector": { "sidecar": false }
 *
 * or tune it:
 *
 *   "connector": { "sidecar": { "crawlerDbPath": "~/.slacrawl/slacrawl.db", "semantic": false } }
 *
 * The sidecar's own database always lives under the AutoRAG workspace, one file
 * per datasource instance, never next to (or inside) the crawler's archive.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, resolve, sep } from "node:path";
import { parse as parseToml } from "smol-toml";
import type { Embedder } from "../../embedding/embedder.ts";
import { sanitizeIdSegment } from "../connector.ts";
import type { CrawlerDescriptor } from "./descriptors.ts";
import type { CrawlerSidecarOptions } from "./index.ts";

export interface ResolveSidecarInput {
	readonly descriptor: CrawlerDescriptor;
	readonly connector: Record<string, unknown> | undefined;
	readonly workspaceRoot: string | undefined;
	readonly registrationName: string;
	readonly instanceId: string;
	/** Where the crawler keeps its archive when the user did not say. */
	readonly defaultCrawlerDb: string;
	/** The embedder shared with the rest of AutoRAG. Without one the sidecar is lexical-only. */
	readonly embedder?: Embedder;
}

export function resolveSidecarOptions(input: ResolveSidecarInput): CrawlerSidecarOptions | undefined {
	const { connector, workspaceRoot } = input;
	const setting = connector?.sidecar;
	if (setting === false || workspaceRoot === undefined) return undefined;
	const object =
		typeof setting === "object" && setting !== null && !Array.isArray(setting)
			? (setting as Record<string, unknown>)
			: {};
	if (object.enabled === false) return undefined;
	const configured = [object.crawlerDbPath, connector?.databasePath].find(
		(value): value is string => typeof value === "string" && value.length > 0,
	);
	const crawlerDbPath = expandHome(configured ?? archivePathFromCrawlerConfig(input) ?? input.defaultCrawlerDb);
	const workspaceId =
		typeof connector?.workspace === "string" && connector.workspace.length > 0 ? connector.workspace : undefined;
	return {
		descriptor: input.descriptor,
		crawlerDbPath,
		sidecarDbPath: join(
			workspaceRoot,
			".autorag",
			"datasources",
			sanitizeIdSegment(input.registrationName),
			sanitizeIdSegment(input.instanceId),
			"sidecar.sqlite",
		),
		...(workspaceId === undefined ? {} : { workspaceId }),
		...(object.semantic === false || input.embedder === undefined ? {} : { embedder: input.embedder }),
	};
}

/**
 * The archive a crawler's own config relocates to (`db_path`), when its descriptor says the config can.
 * Relative paths resolve against the config's directory. Any problem (no configPath, unreadable file,
 * malformed TOML, no `db_path`) means "use the default archive", never an error.
 */
function archivePathFromCrawlerConfig(input: ResolveSidecarInput): string | undefined {
	const configPath = input.connector?.configPath;
	if (input.descriptor.archivePathInConfig !== true || typeof configPath !== "string" || configPath.length === 0) {
		return undefined;
	}
	const file = expandHome(configPath);
	try {
		const dbPath = parseToml(readFileSync(file, "utf8")).db_path;
		if (typeof dbPath !== "string" || dbPath.length === 0) return undefined;
		const expanded = expandHome(dbPath);
		return isAbsolute(expanded) ? expanded : resolve(dirname(file), expanded);
	} catch {
		return undefined;
	}
}

/** Expands a leading `~` or `~/` (`~\` on Windows) to the home directory, joined with platform separators. */
function expandHome(path: string): string {
	if (path === "~") return homedir();
	const prefixed = path.startsWith("~/") || (sep === "\\" && path.startsWith("~\\"));
	return prefixed ? join(homedir(), path.slice(2)) : path;
}
