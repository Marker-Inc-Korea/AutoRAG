import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { AutoRAGAgent } from "../agent/agent.ts";
import { toToolDefinition } from "../agent/pi-session.ts";
import { createSearchAllDocumentsTool } from "../agent/search-all-tool.ts";
import {
	createSingleDatasourceSearchTools,
	type SingleDatasourceToolSpec,
} from "../agent/search-single-datasource-tool.ts";
import { buildAgentOptions, type CliConfig } from "../cli/config.ts";

/** Instance roots shown in a datasource tool description; matches the AutoRAG agent's own cap. */
const MAX_INSTANCE_SCOPES = 8;

/** Which of AutoRAG's retrieval tools a profile session is offered, on top of its own tools. */
export interface RetrievalToolOptions {
	/** `search_all_documents`: fan-out across every configured retrieval method. */
	readonly searchAll?: boolean;
	/** Datasource ids to offer one `search_datasource_<id>` tool for each. */
	readonly datasources?: readonly string[];
}

export interface RetrievalToolkit {
	readonly tools: readonly ToolDefinition[];
	dispose(): Promise<void>;
}

export function hasRetrievalTools(options: RetrievalToolOptions | undefined): boolean {
	return options !== undefined && (options.searchAll === true || (options.datasources?.length ?? 0) > 0);
}

/**
 * Retrieval tools for a profile session, backed by the configured indexes.
 *
 * Retrieval memory is never connected: the backing agent is pointed at a
 * throwaway memory file with the memory embedder off, so nothing the profile
 * searches is read from or recorded into the user's real `memory.json`. Only
 * the search tools are handed out; the backing agent's other surfaces (web,
 * Everything, FSearch, Jev, duplicate scan) stay off.
 */
export async function createRetrievalToolkit(
	config: CliConfig,
	options: RetrievalToolOptions,
): Promise<RetrievalToolkit> {
	if (!hasRetrievalTools(options)) return { tools: [], dispose: async () => {} };

	const scratch = await mkdtemp(join(tmpdir(), "autorag-profile-retrieval-"));
	const { selfConfig: _selfConfig, ...agentOptions } = buildAgentOptions(config);
	const agent = new AutoRAGAgent({
		...agentOptions,
		memoryPath: join(scratch, "memory.json"),
		memoryEmbedder: false,
		webSearch: false,
		everything: false,
		fsearch: false,
		dupey: false,
		jev: false,
	});
	try {
		const catalog = agent.listDatasources();
		const wanted = [...new Set(options.datasources ?? [])];
		const unknown = wanted.filter((id) => !catalog.some((entry) => entry.datasourceId === id));
		if (unknown.length > 0) {
			const known = catalog.map((entry) => entry.datasourceId).join(", ") || "none configured";
			throw new Error(`Unknown datasource id(s) for retrieval tools: ${unknown.join(", ")} (configured: ${known})`);
		}
		const specs: SingleDatasourceToolSpec[] = catalog
			.filter((entry) => wanted.includes(entry.datasourceId))
			.map((entry) => ({
				datasourceId: entry.datasourceId,
				description: entry.description,
				instanceScopes: entry.sourceScopes
					.filter((source) => source.split("/").filter((segment) => segment.length > 0).length === 2)
					.slice(0, MAX_INSTANCE_SCOPES),
			}));
		const tools = [
			...(options.searchAll === true ? [createSearchAllDocumentsTool(agent)] : []),
			...createSingleDatasourceSearchTools(agent, specs),
		].map((tool) => toToolDefinition(tool));
		return {
			tools,
			dispose: async () => {
				await rm(scratch, { recursive: true, force: true });
			},
		};
	} catch (error) {
		await rm(scratch, { recursive: true, force: true });
		throw error;
	}
}
