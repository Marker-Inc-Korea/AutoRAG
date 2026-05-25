import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { PosixRetrieval } from "../retrieval/posix.ts";
import { RetrievalMethodRegistry } from "../retrieval/registry.ts";
import type { NumberedResult, RetrievalOptions } from "../retrieval/types.ts";

export interface AutoRAGToolOptions {
	searchPaths: string[];
	manifestDir?: string;
	memoryPath?: string;
}

export interface AutoRAGToolDetails {
	resultCount: number;
	methodsUsed: string[];
	elapsedMs: number;
	numberedResults: NumberedResult[];
}

const autoragToolSchema = Type.Object({
	query: Type.String({ description: "Search query — text pattern, regex, or glob (e.g. **/*.ts)" }),
	topK: Type.Optional(Type.Number({ description: "Maximum number of results to return (default: 20)" })),
	scope: Type.Optional(Type.String({ description: "Directory to restrict search to" })),
	methods: Type.Optional(
		Type.Array(Type.String(), { description: "Specific retrieval method names to use (default: all active)" }),
	),
});

export function createAutoRAGTool(
	options: AutoRAGToolOptions,
): AgentTool<typeof autoragToolSchema, AutoRAGToolDetails> {
	const registry = new RetrievalMethodRegistry();
	registry.register(new PosixRetrieval({ defaultScope: options.searchPaths[0] ?? process.cwd() }));

	return {
		name: "autorag_search",
		label: "AutoRAG Search",
		description:
			"Search through documents using multiple retrieval methods. Supports grep, glob, vector search, BM25, and more.",
		parameters: autoragToolSchema,
		async execute(
			_toolCallId: string,
			params: { query: string; topK?: number; scope?: string; methods?: string[] },
			signal?: AbortSignal,
		): Promise<AgentToolResult<AutoRAGToolDetails>> {
			const start = Date.now();
			const topK = params.topK ?? 20;

			let activeMethods = registry.list();
			if (params.methods && params.methods.length > 0) {
				activeMethods = activeMethods.filter((m) => params.methods!.includes(m.describe().name));
			}

			const retrievalOptions: RetrievalOptions = { topK, scope: params.scope, signal };
			const allResults: Array<{ source: string; content: string; method: string }> = [];
			const methodsUsed: string[] = [];

			for (const method of activeMethods) {
				const name = method.describe().name;
				try {
					const results = await method.retrieve(params.query, retrievalOptions);
					for (const r of results) {
						allResults.push({ source: r.source, content: r.content, method: name });
					}
					if (results.length > 0) methodsUsed.push(name);
				} catch {
					// stub methods throw NotImplementedError — skip silently
				}
			}

			const numbered: NumberedResult[] = allResults.map((r, i) => ({
				index: i + 1,
				source: r.source,
				content: r.content,
				method: r.method,
			}));
			const text =
				numbered.length === 0
					? "No results found."
					: numbered.map((r) => `[${r.index}] ${r.source} (${r.method})\n    ${r.content}`).join("\n\n");

			return {
				content: [{ type: "text", text }],
				details: {
					resultCount: allResults.length,
					methodsUsed,
					elapsedMs: Date.now() - start,
					numberedResults: numbered,
				},
			};
		},
	};
}
