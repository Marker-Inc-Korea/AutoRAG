import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { NumberedResult } from "../retrieval/types.ts";
import { createAutoRAGTool } from "../tool/tool.ts";

function registryPathFromMemory(memoryPath: string): string {
	return memoryPath.replace(/\.json$/, ".last-results.json");
}

interface CliArgs {
	command: string;
	query: string;
	numbers: number[];
	useful: boolean;
	notUseful: boolean;
	scope: string;
	topK: number;
	format: "text" | "json";
	method: string | undefined;
	manifestDir: string | undefined;
	memoryPath: string;
	help: boolean;
	version: boolean;
}

function parseArgs(argv: string[]): CliArgs {
	const args = argv.slice(2);
	const result: CliArgs = {
		command: "",
		query: "",
		numbers: [],
		useful: false,
		notUseful: false,
		scope: process.cwd(),
		topK: 20,
		format: "text",
		method: undefined,
		manifestDir: undefined,
		memoryPath: join(homedir(), ".autorag", "memory.json"),
		help: false,
		version: false,
	};

	if (args.includes("--help") || args.includes("-h")) {
		result.help = true;
		return result;
	}
	if (args.includes("--version") || args.includes("-v")) {
		result.version = true;
		return result;
	}

	let i = 0;
	while (i < args.length) {
		const arg = args[i];
		if (arg === "search") {
			result.command = "search";
			if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
				i++;
				result.query = args[i];
			}
		} else if (arg === "feedback") {
			result.command = "feedback";
			if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
				i++;
				result.numbers = args[i].split(",").map((n) => Number.parseInt(n.trim(), 10));
			}
		} else if (arg === "--useful") {
			result.useful = true;
		} else if (arg === "--not-useful") {
			result.notUseful = true;
		} else if (arg === "--scope" && i + 1 < args.length) {
			i++;
			result.scope = args[i];
		} else if (arg === "--top-k" && i + 1 < args.length) {
			i++;
			result.topK = Number.parseInt(args[i], 10);
		} else if (arg === "--format" && i + 1 < args.length) {
			i++;
			result.format = args[i] as "text" | "json";
		} else if (arg === "--method" && i + 1 < args.length) {
			i++;
			result.method = args[i];
		} else if (arg === "--manifest-dir" && i + 1 < args.length) {
			i++;
			result.manifestDir = args[i];
		} else if (arg === "--memory-path" && i + 1 < args.length) {
			i++;
			result.memoryPath = args[i];
		}
		i++;
	}

	return result;
}

function printHelp(): void {
	process.stdout.write(`autorag - Self-evolving librarian agent

Usage:
  autorag search <query> [options]
  autorag feedback <numbers> --useful|--not-useful [options]
  autorag --help
  autorag --version

Commands:
  search <query>        Search documents using available retrieval methods
  feedback <numbers>    Submit feedback on search results by number (e.g. 1,3,5)

Search Options:
  --scope <path>        Directory to search in (default: current directory)
  --top-k <n>           Maximum results to return (default: 20)
  --format <text|json>  Output format (default: text)
  --method <name>       Force specific retrieval method
  --manifest-dir <path> Directory with store manifests

Feedback Options:
  --useful              Mark referenced results as useful
  --not-useful          Mark referenced results as not useful

Common Options:
  --memory-path <path>  Path to memory JSON file
  --help, -h            Show this help message
  --version, -v         Show version
`);
}

async function runSearch(args: CliArgs): Promise<void> {
	if (!args.query) {
		process.stderr.write("Error: search requires a query argument\n");
		process.exit(1);
	}

	const tool = createAutoRAGTool({
		searchPaths: [args.scope],
		manifestDir: args.manifestDir,
		memoryPath: args.memoryPath,
	});

	const params: { query: string; topK: number; scope: string; methods?: string[] } = {
		query: args.query,
		topK: args.topK,
		scope: args.scope,
	};
	if (args.method) {
		params.methods = [args.method];
	}

	const result = await tool.execute("cli", params);
	try {
		const registryPath = registryPathFromMemory(args.memoryPath);
		const numberedResults = result.details.numberedResults ?? [];
		const { dirname } = await import("node:path");
		const { mkdirSync } = await import("node:fs");
		const dir = dirname(registryPath);
		if (!existsSync(dir)) {
			mkdirSync(dir, { recursive: true });
		}
		writeFileSync(registryPath, JSON.stringify(numberedResults), "utf-8");
	} catch {
		// Registry save is best-effort — search results are still displayed
	}
	const text = (result.content[0] as { type: "text"; text: string }).text;

	if (args.format === "json") {
		const jsonOutput = {
			results: text === "No results found." ? [] : text.split("\n").map((line) => ({ content: line })),
			metadata: result.details,
		};
		process.stdout.write(`${JSON.stringify(jsonOutput, null, 2)}\n`);
	} else {
		if (text === "No results found.") {
			process.stdout.write("No results found.\n");
		} else {
			process.stdout.write(`${text}\n`);
		}
	}
}

async function runFeedback(args: CliArgs): Promise<void> {
	if (args.numbers.length === 0) {
		process.stderr.write("Error: feedback requires result numbers (e.g. feedback 1,3,5)\n");
		process.exit(1);
	}
	if (!args.useful && !args.notUseful) {
		process.stderr.write("Error: specify --useful or --not-useful\n");
		process.exit(1);
	}
	const registryPath = registryPathFromMemory(args.memoryPath);
	if (!existsSync(registryPath)) {
		process.stderr.write("Error: no previous search results found. Run 'autorag search' first.\n");
		process.exit(1);
	}
	const numberedResults: NumberedResult[] = JSON.parse(readFileSync(registryPath, "utf-8"));
	const resultMap = new Map(numberedResults.map((r) => [r.index, r]));
	const feedback: Array<{ source: string; useful: boolean }> = [];
	const unknown: number[] = [];
	for (const n of args.numbers) {
		const entry = resultMap.get(n);
		if (entry) {
			feedback.push({ source: entry.source, useful: args.useful });
		} else {
			unknown.push(n);
		}
	}
	if (unknown.length > 0) {
		process.stderr.write(`Warning: unknown result numbers: ${unknown.join(", ")}\n`);
	}
	if (feedback.length > 0) {
		const { RetrievalMemory } = await import("../memory/memory.ts");
		const memory = new RetrievalMemory({ storagePath: args.memoryPath });
		memory.load();
		memory.recordResultFeedback(feedback);
		memory.save();
		process.stdout.write(`Recorded feedback for ${feedback.length} result(s).\n`);
	}
}

async function main(): Promise<void> {
	const args = parseArgs(process.argv);

	if (args.help) {
		printHelp();
		return;
	}

	if (args.version) {
		process.stdout.write("0.1.0\n");
		return;
	}

	if (args.command === "search") {
		await runSearch(args);
		return;
	}

	if (args.command === "feedback") {
		await runFeedback(args);
		return;
	}

	printHelp();
}

main().catch((err) => {
	process.stderr.write(`Error: ${(err as Error).message}\n`);
	process.exit(1);
});
