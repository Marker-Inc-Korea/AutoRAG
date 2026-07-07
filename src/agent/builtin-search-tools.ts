import { closeSync, fstatSync, openSync, readdirSync, readSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { planSourceRoots, resolveVirtualSource, type SourceRoot } from "../filesystem/source-paths.ts";

/**
 * AutoRAG-owned, read-only, path-opaque built-in tools surfaced to the agent
 * during `searchDocuments()`. These tools are ALWAYS available and CANNOT be
 * disabled or shadowed by caller-provided `options.tools`: agent.ts reserves
 * their names and merges them last-wins. All tool content and details expose
 * ONLY opaque virtual source ids (e.g. `/docs/notes.txt`); real filesystem
 * paths never leave this module.
 */

export const GREP_TOOL_NAME = "grep";
export const FIND_TOOL_NAME = "find";
export const READ_TOOL_NAME = "read";
export const LS_TOOL_NAME = "ls";
export const STAT_TOOL_NAME = "stat";

export const READ_DEFAULT_MAX_LINES = 200;
export const READ_DEFAULT_MAX_BYTES = 65536;
export const FIND_MAX_RESULTS = 100;

/** Reserved built-in tool names that caller `options.tools` cannot shadow. */
export const BUILTIN_SEARCH_TOOL_NAMES: readonly string[] = [
	GREP_TOOL_NAME,
	FIND_TOOL_NAME,
	READ_TOOL_NAME,
	LS_TOOL_NAME,
	STAT_TOOL_NAME,
] as const;

export interface BuiltinSearchToolsOptions {
	/** Workspace project root, used only for sanitizer context internally. */
	readonly root: string;
	/** Configured search paths; each becomes an opaque source root. */
	readonly searchPaths: readonly string[];
}

interface VirtualEntry {
	readonly sourceId: string;
	readonly realPath: string;
	readonly root: SourceRoot;
}

interface GrepMatch {
	readonly sourceId: string;
	readonly lineNumber: number;
	readonly line: string;
}

interface GrepDetails {
	readonly method: "grep";
	readonly matchCount: number;
	readonly sources: readonly string[];
	readonly truncated: boolean;
}

interface FindDetails {
	readonly method: "find";
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly truncated: boolean;
}

interface ReadDetails {
	readonly method: "read";
	readonly source: string;
	readonly lines: number;
	readonly bytes: number;
	readonly truncated: boolean;
}

interface LsDetails {
	readonly method: "ls";
	readonly entries: readonly string[];
}

interface StatDetails {
	readonly method: "stat";
	readonly source: string;
	readonly kind: "file" | "directory" | "unknown";
	readonly size: number;
}

const grepSchema = Type.Object({
	pattern: Type.String({ description: "Regex or literal string to search for." }),
	isRegex: Type.Optional(Type.Boolean({ description: "Treat pattern as a regex (default false, literal match)." })),
	scope: Type.Optional(
		Type.String({ description: "Optional opaque virtual-path scope, e.g. /docs or /docs/notes.txt." }),
	),
	maxResults: Type.Optional(Type.Integer({ description: "Maximum matches to return." })),
});

const findSchema = Type.Object({
	query: Type.String({ description: "Glob or path substring to match against opaque source ids." }),
});

const readSchema = Type.Object({
	source: Type.String({ description: "Opaque source id to read, e.g. /docs/notes.txt." }),
	start: Type.Optional(Type.Integer({ description: "1-based start line." })),
	end: Type.Optional(Type.Integer({ description: "1-based end line (inclusive)." })),
	maxLines: Type.Optional(Type.Integer({ description: "Maximum lines to return." })),
	maxBytes: Type.Optional(Type.Integer({ description: "Maximum bytes to return." })),
});

const lsSchema = Type.Object({
	source: Type.Optional(
		Type.String({ description: "Opaque directory source id, e.g. /docs. Defaults to all roots." }),
	),
});

const statSchema = Type.Object({
	source: Type.String({ description: "Opaque source id to inspect, e.g. /docs/notes.txt." }),
});

const MAX_GREP_RESULTS = 200;
const MAX_GREP_LINE_LENGTH = 1000;
const BINARY_PROBE_BYTES = 8000;

export function createBuiltinSearchTools(options: BuiltinSearchToolsOptions): AgentTool[] {
	const roots = planSourceRoots(options.searchPaths);

	const resolveEntry = (virtual: string): VirtualEntry | undefined => {
		const resolved = resolveVirtualSource(virtual, roots);
		if (resolved === undefined) return undefined;
		return { sourceId: resolved.sourceId, realPath: resolved.realPath, root: resolved.root };
	};

	const walkAll = (): VirtualEntry[] => {
		const entries: VirtualEntry[] = [];
		for (const root of roots) {
			let rootReal: string;
			try {
				rootReal = realpathSync(root.rootPath);
			} catch {
				continue;
			}
			walkInto(rootReal, root, root.prefix, entries, new Set());
		}
		return entries;
	};

	const listChildren = (dirEntry: VirtualEntry): VirtualEntry[] => {
		const out: VirtualEntry[] = [];
		let names: string[];
		try {
			names = readdirSync(dirEntry.realPath);
		} catch {
			return out;
		}
		for (const name of names) {
			const childReal = resolveChild(dirEntry.realPath, name);
			if (childReal === undefined) continue;
			const childId = `${dirEntry.sourceId === "/" ? "" : dirEntry.sourceId}/${name}`;
			out.push({ sourceId: childId, realPath: childReal, root: dirEntry.root });
		}
		return out;
	};

	return [
		createGrepTool(roots, resolveEntry, walkAll),
		createFindTool(roots, walkAll),
		createReadTool(roots, resolveEntry),
		createLsTool(roots, resolveEntry, listChildren),
		createStatTool(roots, resolveEntry),
	];
}

function createGrepTool(
	_roots: readonly SourceRoot[],
	resolveEntry: (virtual: string) => VirtualEntry | undefined,
	walkAll: () => VirtualEntry[],
): AgentTool<typeof grepSchema, GrepDetails> {
	return {
		name: GREP_TOOL_NAME,
		label: "Grep (AutoRAG built-in)",
		description:
			"Read-only regex/literal line search across configured opaque sources. Results reference only opaque source ids.",
		parameters: grepSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<GrepDetails>> {
			const pattern = String(params.pattern ?? "");
			if (pattern.length === 0) {
				return {
					content: [{ type: "text", text: "Grep pattern was empty; no sources searched." }],
					details: { method: "grep", matchCount: 0, sources: [], truncated: false },
				};
			}
			const isRegex = params.isRegex === true;
			let matcher: RegExp;
			try {
				matcher = isRegex ? new RegExp(pattern, "u") : new RegExp(escapeRegExp(pattern), "u");
			} catch {
				return {
					content: [{ type: "text", text: "Grep pattern was invalid; no sources searched." }],
					details: { method: "grep", matchCount: 0, sources: [], truncated: false },
				};
			}
			const limit = clampPositive(params.maxResults, MAX_GREP_RESULTS, MAX_GREP_RESULTS);
			let targets: VirtualEntry[];
			if (params.scope !== undefined && params.scope.length > 0) {
				const scoped = resolveEntry(params.scope);
				if (scoped === undefined) {
					return outOfScopeResult<GrepDetails>({
						method: "grep",
						matchCount: 0,
						sources: [],
						truncated: false,
					});
				}
				let isDir = false;
				try {
					isDir = statSync(scoped.realPath).isDirectory();
				} catch {
					return outOfScopeResult<GrepDetails>({
						method: "grep",
						matchCount: 0,
						sources: [],
						truncated: false,
					});
				}
				targets = isDir ? collectUnder(scoped) : [scoped];
			} else {
				targets = walkAll();
			}
			const matches: GrepMatch[] = [];
			const matchedSources = new Set<string>();
			let truncated = false;
			for (const target of targets) {
				if (matches.length >= limit) {
					truncated = true;
					break;
				}
				if (!isLikelyText(target.realPath)) {
					continue;
				}
				const fileMatches = grepFile(target, matcher, limit - matches.length);
				for (const m of fileMatches) {
					matches.push(m);
					matchedSources.add(m.sourceId);
					if (matches.length >= limit) {
						truncated = true;
						break;
					}
				}
			}
			const text =
				matches.length === 0
					? "No grep matches."
					: `Grep results:\n\n${matches
							.map((m, i) => `[${i + 1}] ${m.sourceId}:${m.lineNumber} ${m.line}`)
							.join("\n")}${truncated ? `\n\n(truncated at ${matches.length} matches)` : ""}`;
			return {
				content: [{ type: "text", text }],
				details: {
					method: "grep",
					matchCount: matches.length,
					sources: [...matchedSources],
					truncated,
				},
			};
		},
	};
}

function createFindTool(
	_roots: readonly SourceRoot[],
	walkAll: () => VirtualEntry[],
): AgentTool<typeof findSchema, FindDetails> {
	return {
		name: FIND_TOOL_NAME,
		label: "Find (AutoRAG built-in)",
		description: "Read-only glob/substring search over opaque source ids. Caps at 100 results. No real paths.",
		parameters: findSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<FindDetails>> {
			const query = String(params.query ?? "").trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Find query was empty; no sources matched." }],
					details: { method: "find", resultCount: 0, sources: [], truncated: false },
				};
			}
			const all = walkAll();
			const matcher = buildGlobMatcher(query);
			const hits: string[] = [];
			let truncated = false;
			for (const entry of all) {
				if (matcher(entry.sourceId)) {
					if (hits.length >= FIND_MAX_RESULTS) {
						truncated = true;
						break;
					}
					hits.push(entry.sourceId);
				}
			}
			hits.sort();
			const text =
				hits.length === 0
					? "No find matches."
					: `Find results:\n\n${hits.map((h, i) => `[${i + 1}] ${h}`).join("\n")}${
							truncated ? `\n\n(truncated at ${FIND_MAX_RESULTS} results)` : ""
						}`;
			return {
				content: [{ type: "text", text }],
				details: {
					method: "find",
					resultCount: hits.length,
					sources: hits,
					truncated,
				},
			};
		},
	};
}

function createReadTool(
	_roots: readonly SourceRoot[],
	resolveEntry: (virtual: string) => VirtualEntry | undefined,
): AgentTool<typeof readSchema, ReadDetails> {
	return {
		name: READ_TOOL_NAME,
		label: "Read (AutoRAG built-in)",
		description: "Read-only file read by opaque source id. Caps lines/bytes; no real paths in output.",
		parameters: readSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<ReadDetails>> {
			const entry = resolveEntry(String(params.source ?? ""));
			if (entry === undefined) {
				return outOfScopeResult<ReadDetails>({
					method: "read",
					source: "",
					lines: 0,
					bytes: 0,
					truncated: false,
				});
			}
			let info: StatsLike;
			try {
				info = statSync(entry.realPath);
			} catch {
				return outOfScopeResult<ReadDetails>({
					method: "read",
					source: entry.sourceId,
					lines: 0,
					bytes: 0,
					truncated: false,
				});
			}
			if (info.isDirectory()) {
				return {
					content: [{ type: "text", text: `Source is a directory, not a readable file: ${entry.sourceId}` }],
					details: { method: "read", source: entry.sourceId, lines: 0, bytes: 0, truncated: false },
				};
			}
			if (!isLikelyText(entry.realPath)) {
				return {
					content: [{ type: "text", text: `Source is binary or unreadable: ${entry.sourceId}` }],
					details: { method: "read", source: entry.sourceId, lines: 0, bytes: 0, truncated: false },
				};
			}
			const maxLines = clampPositive(params.maxLines, READ_DEFAULT_MAX_LINES, READ_DEFAULT_MAX_LINES);
			const maxBytes = clampPositive(params.maxBytes, READ_DEFAULT_MAX_BYTES, READ_DEFAULT_MAX_BYTES);
			const startLine = clampPositive(params.start, 1);
			const endLine = typeof params.end === "number" && params.end >= startLine ? params.end : undefined;
			const { text, lines, bytes, truncated } = readTextFile(entry, startLine, endLine, maxLines, maxBytes);
			const suffix = truncated ? `\n\n(truncated at ${lines} lines / ${bytes} bytes)` : "";
			return {
				content: [{ type: "text", text: `${entry.sourceId}\n${text}${suffix}` }],
				details: { method: "read", source: entry.sourceId, lines, bytes, truncated },
			};
		},
	};
}

function createLsTool(
	roots: readonly SourceRoot[],
	resolveEntry: (virtual: string) => VirtualEntry | undefined,
	listChildren: (dir: VirtualEntry) => VirtualEntry[],
): AgentTool<typeof lsSchema, LsDetails> {
	return {
		name: LS_TOOL_NAME,
		label: "Ls (AutoRAG built-in)",
		description: "Read-only listing of opaque directory children. No real paths.",
		parameters: lsSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<LsDetails>> {
			const dirs: VirtualEntry[] = [];
			if (params.source !== undefined && String(params.source).length > 0) {
				const entry = resolveEntry(String(params.source));
				if (entry === undefined) {
					return outOfScopeResult<LsDetails>({ method: "ls", entries: [] });
				}
				let isDir = false;
				try {
					isDir = statSync(entry.realPath).isDirectory();
				} catch {
					return outOfScopeResult<LsDetails>({ method: "ls", entries: [] });
				}
				if (!isDir) {
					return {
						content: [{ type: "text", text: `Source is a file, not a directory: ${entry.sourceId}` }],
						details: { method: "ls", entries: [] },
					};
				}
				dirs.push(entry);
			} else {
				for (const root of roots) {
					dirs.push({ sourceId: root.prefix, realPath: root.rootPath, root });
				}
			}
			const out: string[] = [];
			for (const dir of dirs) {
				for (const child of listChildren(dir)) {
					let isDir = false;
					try {
						isDir = statSync(child.realPath).isDirectory();
					} catch {
						continue;
					}
					out.push(isDir ? `${child.sourceId}/` : child.sourceId);
				}
			}
			out.sort();
			const text =
				out.length === 0 ? "No entries." : `Entries:\n\n${out.map((e, i) => `[${i + 1}] ${e}`).join("\n")}`;
			return {
				content: [{ type: "text", text }],
				details: { method: "ls", entries: out },
			};
		},
	};
}

function createStatTool(
	_roots: readonly SourceRoot[],
	resolveEntry: (virtual: string) => VirtualEntry | undefined,
): AgentTool<typeof statSchema, StatDetails> {
	return {
		name: STAT_TOOL_NAME,
		label: "Stat (AutoRAG built-in)",
		description: "Read-only opaque source metadata (file/dir, size). No real paths.",
		parameters: statSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<StatDetails>> {
			const entry = resolveEntry(String(params.source ?? ""));
			if (entry === undefined) {
				return outOfScopeResult<StatDetails>({
					method: "stat",
					source: "",
					kind: "unknown",
					size: 0,
				});
			}
			let info: StatsLike;
			try {
				info = statSync(entry.realPath);
			} catch {
				return outOfScopeResult<StatDetails>({
					method: "stat",
					source: entry.sourceId,
					kind: "unknown",
					size: 0,
				});
			}
			const kind: StatDetails["kind"] = info.isDirectory() ? "directory" : info.isFile() ? "file" : "unknown";
			const size = info.isFile() ? info.size : 0;
			const text = `${entry.sourceId}\nkind=${kind}\nsize=${size}`;
			return {
				content: [{ type: "text", text }],
				details: { method: "stat", source: entry.sourceId, kind, size },
			};
		},
	};
}

// --- helpers ---

interface StatsLike {
	isFile(): boolean;
	isDirectory(): boolean;
	readonly size: number;
}

function escapeRegExp(value: string): string {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function clampPositive(value: number | undefined, fallback: number, maximum = Number.MAX_SAFE_INTEGER): number {
	if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) return fallback;
	return Math.min(Math.floor(value), maximum);
}

function resolveChild(parentReal: string, name: string): string | undefined {
	const candidate = resolve(parentReal, name);
	try {
		const real = realpathSync(candidate);
		const rel = relative(parentReal, real);
		if (rel.startsWith("..") || isAbsolute(rel)) return undefined;
		return real;
	} catch {
		return undefined;
	}
}

function walkInto(dirReal: string, root: SourceRoot, sourceId: string, out: VirtualEntry[], seen: Set<string>): void {
	if (seen.has(dirReal)) return;
	seen.add(dirReal);
	let names: string[];
	try {
		names = readdirSync(dirReal);
	} catch {
		return;
	}
	for (const name of names) {
		const childReal = resolveChild(dirReal, name);
		if (childReal === undefined) continue;
		const childId = `${sourceId === "/" ? "" : sourceId}/${name}`;
		const childEntry: VirtualEntry = { sourceId: childId, realPath: childReal, root };
		let info: StatsLike;
		try {
			info = statSync(childReal);
		} catch {
			continue;
		}
		out.push(childEntry);
		if (info.isDirectory()) {
			walkInto(childReal, root, childId, out, seen);
		}
	}
}

function collectUnder(dirEntry: VirtualEntry): VirtualEntry[] {
	const out: VirtualEntry[] = [dirEntry];
	walkInto(dirEntry.realPath, dirEntry.root, dirEntry.sourceId, out, new Set());
	return out;
}

function isLikelyText(realPath: string): boolean {
	let info: StatsLike;
	try {
		info = statSync(realPath);
	} catch {
		return false;
	}
	if (!info.isFile()) return false;
	if (info.size === 0) return true;
	return probeText(realPath);
}

function probeText(realPath: string): boolean {
	let fd: number;
	try {
		fd = openSync(realPath, "r");
	} catch {
		return false;
	}
	try {
		const buf = Buffer.alloc(Math.min(BINARY_PROBE_BYTES, fstatSync(fd).size));
		const bytesRead = readSync(fd, buf, 0, buf.length, 0);
		for (let i = 0; i < bytesRead; i++) {
			if (buf[i] === 0) return false;
		}
		return true;
	} catch {
		return false;
	} finally {
		try {
			closeSync(fd);
		} catch {
			// ignore
		}
	}
}

function grepFile(target: VirtualEntry, matcher: RegExp, remaining: number): GrepMatch[] {
	const matches: GrepMatch[] = [];
	let fd: number;
	try {
		fd = openSync(target.realPath, "r");
	} catch {
		return matches;
	}
	try {
		const buf = Buffer.alloc(64 * 1024);
		let leftover = "";
		let lineNumber = 1;
		let bytesRead = readSync(fd, buf, 0, buf.length, null);
		while (bytesRead > 0) {
			leftover += buf.subarray(0, bytesRead).toString("utf8");
			const lines = leftover.split("\n");
			leftover = lines.pop() ?? "";
			for (const line of lines) {
				if (matcher.test(line)) {
					matches.push({
						sourceId: target.sourceId,
						lineNumber,
						line: line.slice(0, MAX_GREP_LINE_LENGTH),
					});
					if (matches.length >= remaining) return matches;
				}
				lineNumber++;
			}
			bytesRead = readSync(fd, buf, 0, buf.length, null);
		}
		if (leftover.length > 0 && matcher.test(leftover)) {
			matches.push({
				sourceId: target.sourceId,
				lineNumber,
				line: leftover.slice(0, MAX_GREP_LINE_LENGTH),
			});
		}
		return matches;
	} catch {
		return matches;
	} finally {
		try {
			closeSync(fd);
		} catch {
			// ignore
		}
	}
}

function readTextFile(
	entry: VirtualEntry,
	startLine: number,
	endLine: number | undefined,
	maxLines: number,
	maxBytes: number,
): { text: string; lines: number; bytes: number; truncated: boolean } {
	let fd: number;
	try {
		fd = openSync(entry.realPath, "r");
	} catch {
		return { text: "Source is unreadable.", lines: 0, bytes: 0, truncated: false };
	}
	try {
		const buf = Buffer.alloc(64 * 1024);
		let leftover = "";
		let lineNumber = 1;
		let bytesRead = readSync(fd, buf, 0, buf.length, null);
		const out: string[] = [];
		let outLines = 0;
		let outBytes = 0;
		let truncated = false;
		while (bytesRead > 0) {
			leftover += buf.subarray(0, bytesRead).toString("utf8");
			const lines = leftover.split("\n");
			leftover = lines.pop() ?? "";
			for (const line of lines) {
				if (lineNumber < startLine) {
					lineNumber++;
					continue;
				}
				if (endLine !== undefined && lineNumber > endLine) {
					return finalize(out, outLines, outBytes, true);
				}
				if (outLines >= maxLines) {
					return finalize(out, outLines, outBytes, true);
				}
				const candidate = outLines === 0 ? line : `\n${line}`;
				if (outBytes + Buffer.byteLength(candidate, "utf8") > maxBytes) {
					return finalize(out, outLines, outBytes, true);
				}
				out.push(line);
				outLines++;
				outBytes += Buffer.byteLength(candidate, "utf8");
				lineNumber++;
			}
			bytesRead = readSync(fd, buf, 0, buf.length, null);
		}
		if (leftover.length > 0 && lineNumber >= startLine && (endLine === undefined || lineNumber <= endLine)) {
			if (outLines < maxLines && outBytes + Buffer.byteLength(leftover, "utf8") <= maxBytes) {
				out.push(leftover);
				outLines++;
				outBytes += Buffer.byteLength(leftover, "utf8");
			} else {
				truncated = true;
			}
		}
		return finalize(out, outLines, outBytes, truncated);
	} catch {
		return { text: "Source is unreadable.", lines: 0, bytes: 0, truncated: false };
	} finally {
		try {
			closeSync(fd);
		} catch {
			// ignore
		}
	}
}

function finalize(
	out: string[],
	lines: number,
	bytes: number,
	truncated: boolean,
): {
	text: string;
	lines: number;
	bytes: number;
	truncated: boolean;
} {
	return { text: out.join("\n"), lines, bytes, truncated };
}

function buildGlobMatcher(query: string): (sourceId: string) => boolean {
	if (query.includes("*") || query.includes("?")) {
		const re = globToRegExp(query);
		return (id: string) => re.test(id);
	}
	const lower = query.toLowerCase();
	return (id: string) => id.toLowerCase().includes(lower);
}

function globToRegExp(glob: string): RegExp {
	let out = "^";
	for (let i = 0; i < glob.length; i++) {
		const ch = glob[i];
		if (ch === "*") {
			out += ".*";
		} else if (ch === "?") {
			out += ".";
		} else {
			out += escapeRegExp(ch);
		}
	}
	out += "$";
	return new RegExp(out, "i");
}

function outOfScopeResult<T extends { method: string }>(details: T): AgentToolResult<T> {
	return {
		content: [{ type: "text", text: `Source is out of scope or unavailable.` }],
		details,
	};
}
