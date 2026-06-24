import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { ExtensionAPI, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { parseInternalMapping } from "./agent/parse-mapping.ts";
import { buildSystemPrompt } from "./agent/system-prompt.ts";
import { loadManifests } from "./manifest/loader.ts";
import { RetrievalMemory } from "./memory/memory.ts";
import { renderMemoryContext } from "./memory/renderer.ts";
import { syncParsedMirrors } from "./mirror/sync.ts";

const ACTIVE_TOOLS = ["grep", "find", "read", "ls", "check_memory", "bash"] as const;

function firstText(event: ToolResultEvent): string {
	return event.content
		.filter((content) => content.type === "text")
		.map((content) => content.text)
		.join("\n");
}

function resultCount(text: string): number {
	const trimmed = text.trim();
	if (!trimmed || trimmed === "No matches found" || trimmed === "No results found.") return 0;
	return trimmed.split("\n").filter(Boolean).length;
}

function queryFromInput(input: Record<string, unknown>): string {
	for (const key of ["pattern", "query", "glob", "name", "path"]) {
		const value = input[key];
		if (typeof value === "string" && value.length > 0) return value;
	}
	return "unknown";
}

function loadSources(cwd: string): string[] {
	const sourcesPath = join(cwd, ".autorag", "sources.json");
	if (!existsSync(sourcesPath)) return [];
	try {
		const parsed = JSON.parse(readFileSync(sourcesPath, "utf8"));
		return Array.isArray(parsed) ? parsed.filter((p): p is string => typeof p === "string") : [];
	} catch {
		return [];
	}
}

export default function autoragExtension(pi: ExtensionAPI): void {
	let memory: RetrievalMemory | undefined;
	let cwd = process.cwd();
	let sources: string[] = [];

	pi.registerTool({
		name: "check_memory",
		label: "Check Memory",
		description:
			"Query past search outcomes before searching. Returns which methods succeeded or failed for similar queries.",
		promptSnippet: "Check past search outcomes before searching",
		promptGuidelines: [
			"Call check_memory before executing a search to see which methods and queries succeeded or failed in past sessions.",
			"Memory is advisory — it reflects past outcomes, not guarantees.",
		],
		parameters: Type.Object({
			query: Type.String({ description: "The query you plan to search for" }),
		}),
		async execute(_toolCallId, params) {
			if (!memory) {
				return { content: [{ type: "text", text: "Memory not initialized yet." }], details: undefined };
			}

			const entries = memory.getEntries();
			const summary = renderMemoryContext(entries);
			const priority = memory.getMethodPriority(params.query);
			const recommendation =
				priority.length > 0
					? "\n\n## Recommended Methods\n" +
						priority
							.map((p, i) => `${i + 1}. **${p.method}** (usefulness: ${(p.score * 100).toFixed(0)}%)`)
							.join("\n")
					: "";

			return {
				content: [{ type: "text", text: summary + recommendation }],
				details: {
					entryCount: entries.length,
					topMethod: priority[0]?.method ?? null,
				},
			};
		},
	});

	pi.registerCommand("autorag-refresh", {
		description: "Re-parse supported files from configured source directories into .autorag/parsed.",
		async handler() {
			const parsed = await syncParsedMirrors({ root: cwd, searchPaths: sources, force: true });
			pi.appendEntry("autorag_refresh", { parsed, timestamp: Date.now() });
		},
	});

	pi.registerCommand("autorag-parse", {
		description: "Parse supported source files into safe markdown mirrors under .autorag/parsed.",
		async handler() {
			const parsed = await syncParsedMirrors({ root: cwd, searchPaths: sources });
			pi.appendEntry("autorag_parse", { parsed, timestamp: Date.now() });
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		cwd = ctx.cwd;
		const memoryPath = join(cwd, ".autorag", "memory.json");
		mkdirSync(dirname(memoryPath), { recursive: true });
		memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		sources = loadSources(cwd);
		await syncParsedMirrors({ root: cwd, searchPaths: sources });
	});

	pi.on("tool_result", async (event) => {
		if (!memory) return;
		if (event.toolName !== "grep" && event.toolName !== "find") return;

		const text = firstText(event);
		memory.append({
			query: queryFromInput(event.input),
			method: event.toolName,
			outcome: "pending",
			metadata: { resultCount: resultCount(text) },
		});
		memory.save();
	});

	pi.on("before_agent_start", async (event, ctx) => {
		if (!memory) {
			const memoryPath = join(ctx.cwd, ".autorag", "memory.json");
			mkdirSync(dirname(memoryPath), { recursive: true });
			memory = new RetrievalMemory({ storagePath: memoryPath });
			memory.load();
		}

		pi.setActiveTools([...ACTIVE_TOOLS]);

		const manifests = loadManifests(join(ctx.cwd, ".autorag", "manifests"));
		const systemPrompt = buildSystemPrompt({
			mode: "extension",
			toolNames: [...ACTIVE_TOOLS],
			memoryEntries: memory.getEntries(),
			manifests,
		});
		const memorySummary =
			memory.getEntries().length > 0
				? `\n\n<memory_context>\n${renderMemoryContext(memory.getEntries())}\n</memory_context>`
				: "";
		return { systemPrompt: `${event.systemPrompt}\n\n${systemPrompt}${memorySummary}` };
	});

	pi.on("message_end", async (event) => {
		if (event.message.role !== "assistant") return;
		const text = event.message.content
			.filter((content) => content.type === "text")
			.map((content) => content.text)
			.join("\n");
		const mapped = parseInternalMapping(text);
		if (mapped.length > 0) {
			pi.appendEntry("autorag_memory", { cwd, mapped, timestamp: Date.now() });
		}
	});
}
