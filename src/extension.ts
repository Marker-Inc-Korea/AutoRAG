import { spawn } from "node:child_process";
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
const JIKJI_MEDIA_ENV_KEY = "JIKJI_ENABLE_MEDIA_INDEX";
const JIKJI_PREPARE_TIMEOUT_MS = 10_000;

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
interface JikjiRefreshConfig {
	readonly enabled: true;
	readonly binaryPath?: string;
	readonly includeHidden?: boolean;
	readonly includeSensitive?: boolean;
	readonly parseTimeout?: number;
	readonly maxFiles?: number;
	readonly staleAfterSeconds?: number;
	readonly timeoutMs?: number;
	readonly exclude?: readonly string[];
}

function loadJikjiConfig(cwd: string): JikjiRefreshConfig | undefined {
	const configPath = join(cwd, ".autorag", "jikji.json");
	if (!existsSync(configPath)) return undefined;
	try {
		const parsed: unknown = JSON.parse(readFileSync(configPath, "utf8"));
		if (!isRecord(parsed) || parsed.enabled !== true) return undefined;
		return {
			enabled: true,
			binaryPath: typeof parsed.binaryPath === "string" ? parsed.binaryPath : undefined,
			includeHidden: parsed.includeHidden === true,
			includeSensitive: parsed.includeSensitive === true,
			parseTimeout: numberOption(parsed.parseTimeout),
			maxFiles: numberOption(parsed.maxFiles),
			staleAfterSeconds: numberOption(parsed.staleAfterSeconds),
			timeoutMs: numberOption(parsed.timeoutMs),
			exclude: stringArrayOption(parsed.exclude),
		};
	} catch {
		return undefined;
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberOption(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function stringArrayOption(value: unknown): readonly string[] | undefined {
	if (value === undefined) return undefined;
	if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) return undefined;
	return value;
}

function prepareJikjiSource(
	config: JikjiRefreshConfig,
	source: string,
): Promise<{ status: "success" | "failed"; source: string; code: number | null }> {
	return new Promise((resolve) => {
		const args = ["prepare", source, "--json"];
		if (config.includeHidden) args.push("--include-hidden");
		if (config.includeSensitive) args.push("--include-sensitive");
		if (config.parseTimeout !== undefined) args.push("--parse-timeout", String(config.parseTimeout));
		if (config.maxFiles !== undefined) args.push("--max-files", String(config.maxFiles));
		if (config.staleAfterSeconds !== undefined) args.push("--stale-after-seconds", String(config.staleAfterSeconds));
		for (const pattern of config.exclude ?? []) args.push("--exclude", pattern);
		const child = spawn(config.binaryPath ?? "jikji", args, {
			env: controlledJikjiEnv(),
			stdio: ["ignore", "ignore", "ignore"],
		});
		let settled = false;
		const timeout = setTimeout(() => {
			if (settled) return;
			settled = true;
			if (!child.killed) child.kill("SIGTERM");
			resolve({ status: "failed", source, code: null });
		}, config.timeoutMs ?? JIKJI_PREPARE_TIMEOUT_MS);
		child.on("error", () => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			resolve({ status: "failed", source, code: null });
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			resolve({ status: code === 0 ? "success" : "failed", source, code });
		});
	});
}

function controlledJikjiEnv(): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (key !== JIKJI_MEDIA_ENV_KEY && value !== undefined) env[key] = value;
	}
	delete env[JIKJI_MEDIA_ENV_KEY];
	return env;
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

	pi.registerCommand("autorag-jikji-refresh", {
		description: "Prepare configured source directories with optional Jikji CLI integration.",
		async handler() {
			const config = loadJikjiConfig(cwd);
			if (!config || sources.length === 0) return;
			const results = [];
			for (const source of sources) {
				results.push(await prepareJikjiSource(config, source));
			}
			pi.appendEntry("autorag_jikji_refresh", { results, timestamp: Date.now() });
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
