/**
 * Turns a datasource's `connector.sidecar` config into {@link CrawlerSidecarOptions}.
 *
 *   "connector": { "sidecar": true }
 *   "connector": { "sidecar": { "crawlerDbPath": "~/.slacrawl/slacrawl.db", "semantic": false } }
 *
 * Off by default. The sidecar's own database always lives under the AutoRAG
 * workspace, one file per datasource instance, never next to (or inside) the
 * crawler's archive.
 */

import { homedir } from "node:os";
import { join } from "node:path";
import { createGatewayEmbedder } from "../../embedding-runtime/gateway-embedder.ts";
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
}

export function resolveSidecarOptions(input: ResolveSidecarInput): CrawlerSidecarOptions | undefined {
	const { connector, workspaceRoot } = input;
	const setting = connector?.sidecar;
	if (setting === undefined || setting === false || workspaceRoot === undefined) return undefined;
	const object =
		typeof setting === "object" && setting !== null && !Array.isArray(setting)
			? (setting as Record<string, unknown>)
			: {};
	if (object.enabled === false) return undefined;
	const configured = [object.crawlerDbPath, connector?.databasePath].find(
		(value): value is string => typeof value === "string" && value.length > 0,
	);
	const crawlerDbPath = (configured ?? input.defaultCrawlerDb).replace(/^~(?=$|\/)/, homedir());
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
		...(object.semantic === false ? {} : { embedder: createGatewayEmbedder() }),
	};
}
