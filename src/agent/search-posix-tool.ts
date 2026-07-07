import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { PosixMethod } from "../retrieval/methods/posix.ts";
import type { RetrievalResult } from "../retrieval/types.ts";

export const SEARCH_POSIX_DOCUMENTS_TOOL_NAME = "search_posix_documents";

const searchPosixSchema = Type.Object({
	query: Type.String({ description: "Regex or literal query to grep across configured source directories." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of posix grep hits to return." })),
	scope: Type.Optional(Type.String({ description: "Optional opaque virtual-path scope, e.g. /docs or /docs/**." })),
});

export interface SearchPosixDocumentsDetails {
	readonly method: "search_posix_documents";
	readonly resultCount: number;
	readonly sources: readonly string[];
}

/**
 * LLM-facing, path-opaque wrapper around the {@link PosixMethod} filesystem
 * grep retrieval. The model can only supply `query`, `topK`, and an opaque
 * `scope`; `result.source` is the opaque source identifier emitted by the
 * method, never a real filesystem path.
 */
export function createSearchPosixDocumentsTool(
	getMethod: () => PosixMethod | undefined,
): AgentTool<typeof searchPosixSchema, SearchPosixDocumentsDetails> {
	return {
		name: SEARCH_POSIX_DOCUMENTS_TOOL_NAME,
		label: "Search Posix Documents",
		description:
			"Search configured source directories with literal/regex grep. Use for exact substring matches, identifiers, and folder-scoped source search.",
		parameters: searchPosixSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchPosixDocumentsDetails>> {
			const method = getMethod();
			if (!method) {
				return unavailableResult("Posix content search is not configured for this AutoRAG agent");
			}
			if (params.query.trim().length === 0) {
				return {
					content: [{ type: "text", text: "Posix query was empty; no documents searched." }],
					details: { method: "search_posix_documents", resultCount: 0, sources: [] },
				};
			}
			try {
				const results = await method.retrieve(params.query, { topK: params.topK, scope: params.scope });
				return {
					content: [{ type: "text", text: formatResults(results) }],
					details: {
						method: "search_posix_documents",
						resultCount: results.length,
						sources: [...new Set(results.map((result) => result.source))],
					},
				};
			} catch (error) {
				return unavailableResult(`Posix search failed: ${errorToPathFreeMessage(error)}`);
			}
		},
	};
}

function unavailableResult(message: string): AgentToolResult<SearchPosixDocumentsDetails> {
	return {
		content: [{ type: "text", text: message }],
		details: { method: "search_posix_documents", resultCount: 0, sources: [] },
	};
}

function formatResults(results: readonly RetrievalResult[]): string {
	if (results.length === 0) return "No posix results.";
	const rows = results.map((result, index) => {
		const line = result.content.replace(/\s+/gu, " ").slice(0, 500);
		return `[${index + 1}] ${result.source} score=${result.score.toFixed(4)}\n${line}`;
	});
	return `Posix results:\n\n${rows.join("\n\n")}`;
}

/** Reduce an error to a path-free summary so the model-facing message never leaks real paths. */
function errorToPathFreeMessage(error: unknown): string {
	const name = error instanceof Error ? error.name : "Error";
	return `${name}; details suppressed for path opacity.`;
}
