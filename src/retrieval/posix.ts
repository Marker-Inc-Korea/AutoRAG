import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "./types.ts";

export interface PosixRetrievalOptions {
	defaultScope: string;
}

function isGlobPattern(query: string): boolean {
	return /[*?[\]{},]/.test(query);
}

export class PosixRetrieval implements RetrievalMethod {
	private readonly defaultScope: string;

	constructor(options: PosixRetrievalOptions) {
		this.defaultScope = options.defaultScope;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "posix",
			type: "posix",
			description: "File system search using ripgrep (rg) for content search and glob pattern matching",
			status: "active",
			capabilities: ["grep", "glob", "regex", "content-search", "file-discovery"],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const { topK = 100, scope, signal } = options;
		const searchPath = scope ? resolve(scope) : resolve(this.defaultScope);

		if (signal?.aborted) {
			throw new Error("Operation aborted");
		}

		if (isGlobPattern(query)) {
			return this.globSearch(query, searchPath, topK, signal);
		}
		return this.contentSearch(query, searchPath, topK, signal);
	}

	private async globSearch(
		pattern: string,
		searchPath: string,
		topK: number,
		signal?: AbortSignal,
	): Promise<RetrievalResult[]> {
		return new Promise((resolvePromise, reject) => {
			if (signal?.aborted) {
				reject(new Error("Operation aborted"));
				return;
			}

			const args = ["--files", "--glob", pattern, searchPath];
			const child = spawn("rg", args, { stdio: ["ignore", "pipe", "pipe"] });
			const lines: string[] = [];
			let stderr = "";
			let aborted = false;

			const onAbort = () => {
				aborted = true;
				child.kill();
			};
			signal?.addEventListener("abort", onAbort, { once: true });

			const rl = createInterface({ input: child.stdout });
			rl.on("line", (line) => {
				if (line.trim() && lines.length < topK) {
					lines.push(line.trim());
				}
			});
			child.stderr?.on("data", (chunk: Buffer) => {
				stderr += chunk.toString();
			});
			child.on("error", (err) => {
				signal?.removeEventListener("abort", onAbort);
				reject(new Error(`Failed to run ripgrep: ${err.message}`));
			});
			child.on("close", (code) => {
				signal?.removeEventListener("abort", onAbort);
				rl.close();
				if (aborted) {
					reject(new Error("Operation aborted"));
					return;
				}
				if (code !== 0 && code !== 1 && lines.length === 0 && code !== null) {
					reject(new Error(stderr.trim() || `ripgrep exited with code ${code}`));
					return;
				}
				const results: RetrievalResult[] = lines.map((filePath, i) => ({
					id: `posix-glob-${i}`,
					content: filePath,
					source: filePath,
					score: 1.0,
					metadata: { type: "file-path", pattern },
				}));
				resolvePromise(results);
			});
		});
	}

	private async contentSearch(
		pattern: string,
		searchPath: string,
		topK: number,
		signal?: AbortSignal,
	): Promise<RetrievalResult[]> {
		return new Promise((resolvePromise, reject) => {
			if (signal?.aborted) {
				reject(new Error("Operation aborted"));
				return;
			}

			const args = ["--json", "--line-number", "--color=never", "--", pattern, searchPath];
			const child = spawn("rg", args, { stdio: ["ignore", "pipe", "pipe"] });
			const matches: Array<{ filePath: string; lineNumber: number; lineText: string }> = [];
			let stderr = "";
			let aborted = false;
			let matchCount = 0;

			const onAbort = () => {
				aborted = true;
				child.kill();
			};
			signal?.addEventListener("abort", onAbort, { once: true });

			const rl = createInterface({ input: child.stdout });
			rl.on("line", (line) => {
				if (!line.trim() || matchCount >= topK) return;
				let event: unknown;
				try {
					event = JSON.parse(line);
				} catch {
					return;
				}
				const ev = event as Record<string, unknown>;
				if (ev["type"] === "match") {
					matchCount++;
					const data = ev["data"] as Record<string, unknown>;
					const pathObj = data["path"] as Record<string, unknown>;
					const filePath = pathObj?.["text"] as string;
					const lineNumber = data["line_number"] as number;
					const linesObj = data["lines"] as Record<string, unknown>;
					const lineText = (linesObj?.["text"] as string) ?? "";
					if (filePath && typeof lineNumber === "number") {
						matches.push({ filePath, lineNumber, lineText: lineText.replace(/\r?\n$/, "") });
					}
					if (matchCount >= topK) {
						child.kill();
					}
				}
			});
			child.stderr?.on("data", (chunk: Buffer) => {
				stderr += chunk.toString();
			});
			child.on("error", (err) => {
				signal?.removeEventListener("abort", onAbort);
				reject(new Error(`Failed to run ripgrep: ${err.message}`));
			});
			child.on("close", (code) => {
				signal?.removeEventListener("abort", onAbort);
				rl.close();
				if (aborted) {
					reject(new Error("Operation aborted"));
					return;
				}
				if (code !== 0 && code !== 1 && matches.length === 0 && code !== null) {
					reject(new Error(stderr.trim() || `ripgrep exited with code ${code}`));
					return;
				}
				const results: RetrievalResult[] = matches.map((m, i) => ({
					id: `posix-grep-${i}`,
					content: `${m.filePath}:${m.lineNumber}: ${m.lineText}`,
					source: m.filePath,
					score: 1.0,
					metadata: { lineNumber: m.lineNumber, pattern },
				}));
				resolvePromise(results);
			});
		});
	}
}
