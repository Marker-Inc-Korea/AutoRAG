import { opendir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../../filesystem/source-paths.ts";
import { matchesVirtualPathScope } from "../scope.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export interface PosixMethodOptions {
	readonly root: string;
	readonly searchPaths: readonly string[];
}

interface SearchFile {
	readonly source: string;
	readonly sourcePath: string;
}

interface PosixHit {
	readonly source: string;
	readonly sourcePath: string;
	readonly line: string;
	readonly lineNumber: number;
	readonly matchCount: number;
	readonly score: number;
}

export class PosixMethod implements RetrievalMethod {
	private readonly root: string;
	private readonly searchPaths: readonly string[];

	constructor(options: PosixMethodOptions) {
		this.root = resolve(options.root);
		this.searchPaths = options.searchPaths;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "posix",
			type: "posix",
			description: "real filesystem content search over configured source directories",
			status: "active",
			capabilities: ["regex", "literal", "glob-scope", "opaque-root-relative-paths"],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const regex = compileQuery(query);
		const files = await listFiles(this.root, this.searchPaths, options.scope);
		const hits: PosixHit[] = [];
		for (const file of files) {
			const content = await readFile(file.sourcePath, "utf8");
			const hit = grepFile(file, content, regex);
			if (hit) hits.push(hit);
		}
		hits.sort((a, b) => b.score - a.score);
		const limited = options.topK === undefined ? hits : hits.slice(0, options.topK);
		return limited.map((hit) => ({
			id: `${hit.source}:${hit.lineNumber}`,
			content: hit.line,
			source: hit.source,
			score: hit.score,
			metadata: {
				lineNumber: hit.lineNumber,
				matchCount: hit.matchCount,
				method: "posix",
			},
		}));
	}
}

function compileQuery(query: string): RegExp {
	try {
		return new RegExp(query, "g");
	} catch {
		return new RegExp(escapeRegExp(query), "g");
	}
}

function escapeRegExp(literal: string): string {
	return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function listFiles(
	root: string,
	searchPaths: readonly string[],
	scope: string | undefined,
): Promise<SearchFile[]> {
	const files: SearchFile[] = [];
	for (const sourceRoot of planSourceRoots(searchPaths)) {
		await collectFiles(root, sourceRoot, sourceRoot.rootPath, files, scope);
	}
	files.sort((a, b) => a.source.localeCompare(b.source));
	return files;
}

async function collectFiles(
	root: string,
	sourceRoot: SourceRoot,
	directory: string,
	files: SearchFile[],
	scope: string | undefined,
): Promise<void> {
	const dir = await opendir(directory);
	for await (const entry of dir) {
		if (entry.name === ".autorag" || entry.name === ".git" || entry.name === "node_modules") continue;
		const sourcePath = resolve(directory, entry.name);
		if (entry.isDirectory()) {
			await collectFiles(root, sourceRoot, sourcePath, files, scope);
			continue;
		}
		if (!entry.isFile()) continue;
		const source = opaqueSource(root, sourceRoot, sourcePath);
		if (!matchesVirtualPathScope(source, scope)) continue;
		files.push({ source, sourcePath });
	}
}

function opaqueSource(_root: string, sourceRoot: SourceRoot, sourcePath: string): string {
	return sourceIdentifier(sourceRoot, sourcePath);
}

function grepFile(file: SearchFile, content: string, regex: RegExp): PosixHit | undefined {
	const lines = content.split(/\r?\n/);
	let matchCount = 0;
	let line = "";
	let lineNumber = 0;
	for (let index = 0; index < lines.length; index += 1) {
		regex.lastIndex = 0;
		const matches = lines[index]?.match(regex) ?? [];
		if (matches.length === 0) continue;
		matchCount += matches.length;
		if (lineNumber === 0) {
			line = lines[index] ?? "";
			lineNumber = index + 1;
		}
	}
	if (matchCount === 0) return undefined;
	return {
		source: file.source,
		sourcePath: file.sourcePath,
		line,
		lineNumber,
		matchCount,
		score: matchCount + 1 / (1 + pathDepth(file.source)),
	};
}

function pathDepth(source: string): number {
	return (source.match(/\//g) ?? []).length;
}
