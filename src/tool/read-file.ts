import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";

const MAX_FILE_SIZE = 100 * 1024;

const readFileSchema = Type.Object({
	path: Type.String({ description: "Absolute or relative file path to read" }),
	startLine: Type.Optional(Type.Number({ description: "Start line (1-indexed, inclusive)" })),
	endLine: Type.Optional(Type.Number({ description: "End line (1-indexed, inclusive)" })),
});

export interface ReadFileDetails {
	path: string;
	lineCount: number;
	truncated: boolean;
}

export function createReadFileTool(options: {
	searchPaths: string[];
}): AgentTool<typeof readFileSchema, ReadFileDetails> {
	const resolvedSearchPaths = options.searchPaths.map((p) => resolve(p));

	return {
		name: "read_file",
		label: "Read File",
		description: "Read file contents with optional line range. Returns numbered lines for reference.",
		parameters: readFileSchema,
		async execute(
			_toolCallId: string,
			params: { path: string; startLine?: number; endLine?: number },
		): Promise<AgentToolResult<ReadFileDetails>> {
			const filePath = resolve(params.path);

			const isAllowed = resolvedSearchPaths.some((sp) => filePath.startsWith(sp));
			if (!isAllowed) {
				return {
					content: [{ type: "text", text: `Error: path "${filePath}" is outside allowed search paths.` }],
					details: { path: filePath, lineCount: 0, truncated: false },
				};
			}

			let stat;
			try {
				stat = statSync(filePath);
			} catch {
				return {
					content: [{ type: "text", text: `Error: file not found — ${filePath}` }],
					details: { path: filePath, lineCount: 0, truncated: false },
				};
			}

			if (stat.isDirectory()) {
				return {
					content: [{ type: "text", text: `Error: "${filePath}" is a directory, not a file.` }],
					details: { path: filePath, lineCount: 0, truncated: false },
				};
			}

			let rawContent: Buffer;
			try {
				rawContent = readFileSync(filePath);
			} catch (err) {
				return {
					content: [{ type: "text", text: `Error reading file: ${(err as Error).message}` }],
					details: { path: filePath, lineCount: 0, truncated: false },
				};
			}

			const hasNullByte = rawContent.subarray(0, Math.min(8192, rawContent.length)).includes(0);
			if (hasNullByte) {
				return {
					content: [{ type: "text", text: `Binary file: ${filePath}` }],
					details: { path: filePath, lineCount: 0, truncated: false },
				};
			}

			let truncated = false;
			let textContent: string;
			if (rawContent.length > MAX_FILE_SIZE) {
				textContent = rawContent.subarray(0, MAX_FILE_SIZE).toString("utf-8");
				truncated = true;
			} else {
				textContent = rawContent.toString("utf-8");
			}

			let lines = textContent.split("\n");
			if (lines.length > 0 && lines[lines.length - 1] === "") {
				lines = lines.slice(0, -1);
			}

			const totalLines = lines.length;
			const start = params.startLine ? Math.max(1, params.startLine) : 1;
			const end = params.endLine ? Math.min(totalLines, params.endLine) : totalLines;
			const selectedLines = lines.slice(start - 1, end);

			const numbered = selectedLines.map((line, i) => `${start + i}: ${line}`).join("\n");
			const truncationNote = truncated ? `\n... (truncated, file is ${rawContent.length} bytes)` : "";

			return {
				content: [{ type: "text", text: numbered + truncationNote }],
				details: { path: filePath, lineCount: selectedLines.length, truncated },
			};
		},
	};
}
