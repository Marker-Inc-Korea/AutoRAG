import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import type { Workspace } from "@nomadamas/agentdir";
import { type TSchema, Type } from "typebox";
import { agentdirGrep } from "./grep-core.ts";

/** Names of the agentdir virtual-path tools that replace Pi's builtin grep/find/read/ls + virtual ops. */
export const AGENTDIR_TOOL_NAMES = ["grep", "find", "read", "ls", "stat", "mv", "cp", "mkdir", "rmdir"] as const;

/** Tools whose results are recorded in retrieval memory (search tools only). */
export const SEARCH_TOOLS = ["grep", "find"] as const;

/**
 * The closed agent tool surface. Builtin grep/find/read/ls/bash/edit/write are
 * NOT included, enforcing path opacity. `check_memory` and the organizer
 * delegation tool `organize` are the only non-agentdir entries.
 */
export const ACTIVE_TOOLS = [...AGENTDIR_TOOL_NAMES, "check_memory", "organize"] as const;

/** Uniform details contract so the memory hooks can read method/sources/resultCount. */
export interface AgentdirToolDetails {
	method: string;
	sources: string[];
	resultCount: number;
}

interface ToolSpec<S extends TSchema> {
	name: string;
	label: string;
	description: string;
	guideline: string;
	parameters: S;
	run(ws: Workspace, params: Record<string, unknown>): Promise<{ text: string; details: AgentdirToolDetails }>;
}

function stripTrailingSlash(path: string): string {
	return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

function toGlob(pattern: string): string {
	if (pattern.startsWith("/")) return pattern;
	return `/**/${pattern}`;
}

const grepSpec: ToolSpec<TSchema> = {
	name: "grep",
	label: "Grep (virtual)",
	description:
		"Search file CONTENTS across the virtual document tree by regex/literal. Returns virtual paths and matching lines, ranked by match count.",
	guideline:
		"Use grep to find specific text inside files. Paths are virtual (e.g. /docs/report.md); source locations are never exposed.",
	parameters: Type.Object({
		pattern: Type.String({ description: "Regex or literal text to search for inside files" }),
		path: Type.Optional(Type.String({ description: "Restrict to a virtual glob, e.g. /docs/**/*.md" })),
		ignoreCase: Type.Optional(Type.Boolean({ description: "Case-insensitive match" })),
	}),
	async run(ws, params) {
		const hits = await agentdirGrep(ws, String(params.pattern), {
			pathGlob: typeof params.path === "string" ? params.path : undefined,
			ignoreCase: params.ignoreCase === true,
		});
		const text =
			hits.length === 0
				? "No matches found"
				: hits.map((h) => `${h.virtualPath}:${h.lineNumber}: ${h.line.trim()}`).join("\n");
		return {
			text,
			details: { method: "grep", sources: hits.map((h) => h.virtualPath), resultCount: hits.length },
		};
	},
};

const findSpec: ToolSpec<TSchema> = {
	name: "find",
	label: "Find (virtual)",
	description: "Find FILES by name or glob in the virtual tree. Returns virtual paths only.",
	guideline: "Use find to discover files by name/extension, e.g. pattern '*.pdf'. Results are virtual paths.",
	parameters: Type.Object({
		pattern: Type.String({ description: "Name or glob, e.g. *.md, report.txt, or /docs/**/*.csv" }),
	}),
	async run(ws, params) {
		const matches = await ws.rglob(toGlob(String(params.pattern)));
		const text = matches.length === 0 ? "No matches found" : matches.join("\n");
		return { text, details: { method: "find", sources: matches, resultCount: matches.length } };
	},
};

const readSpec: ToolSpec<TSchema> = {
	name: "read",
	label: "Read (virtual)",
	description: "Read a file's contents by virtual path.",
	guideline: "Use read to examine a file found via grep/find. Pass the virtual path exactly as returned.",
	parameters: Type.Object({
		path: Type.String({ description: "Virtual path of the file to read, e.g. /docs/report.md" }),
	}),
	async run(ws, params) {
		const path = String(params.path);
		try {
			const text = (await ws.readBytes(path)).toString("utf8");
			return { text, details: { method: "read", sources: [path], resultCount: 1 } };
		} catch {
			return {
				text: `Cannot read virtual path: ${path} (not a readable file)`,
				details: { method: "read", sources: [], resultCount: 0 },
			};
		}
	},
};

const lsSpec: ToolSpec<TSchema> = {
	name: "ls",
	label: "Ls (virtual)",
	description: "List entries under a virtual directory.",
	guideline: "Use ls to inspect the virtual tree structure before narrowing a search. Default lists the root.",
	parameters: Type.Object({
		path: Type.Optional(Type.String({ description: "Virtual directory path, defaults to /" })),
	}),
	async run(ws, params) {
		const dir = typeof params.path === "string" && params.path.length > 0 ? stripTrailingSlash(params.path) : "/";
		const glob = dir === "/" ? "/*" : `${dir}/*`;
		const entries = await ws.rglob(glob);
		const text = entries.length === 0 ? "(empty)" : entries.join("\n");
		return { text, details: { method: "ls", sources: entries, resultCount: entries.length } };
	},
};

const statSpec: ToolSpec<TSchema> = {
	name: "stat",
	label: "Stat (virtual)",
	description: "Show metadata for a virtual path (size, mtime, type). Source paths are not exposed.",
	guideline: "Use stat to check a file's size/type via its virtual path.",
	parameters: Type.Object({
		path: Type.String({ description: "Virtual path to stat" }),
	}),
	async run(ws, params) {
		const path = String(params.path);
		const s = await ws.stat(path);
		// Strip sourcePath to preserve path opacity.
		const safe = {
			virtualPath: s.virtualPath,
			sizeBytes: s.sizeBytes,
			mtimeNs: s.mtimeNs,
			entryType: s.entryType,
			materialized: s.materialized,
		};
		return { text: JSON.stringify(safe, null, 2), details: { method: "stat", sources: [path], resultCount: 1 } };
	},
};

const mvSpec: ToolSpec<TSchema> = {
	name: "mv",
	label: "Move (virtual)",
	description: "Move/rename an entry in the virtual namespace. Original source files are NOT modified.",
	guideline: "Use mv to reorganize the virtual layout. Only the virtual tree changes; source files stay put.",
	parameters: Type.Object({
		from: Type.String({ description: "Source virtual path" }),
		to: Type.String({ description: "Destination virtual path" }),
	}),
	async run(ws, params) {
		const from = String(params.from);
		const to = String(params.to);
		await ws.mv(from, to);
		return { text: `Moved ${from} -> ${to}`, details: { method: "mv", sources: [to], resultCount: 1 } };
	},
};

const cpSpec: ToolSpec<TSchema> = {
	name: "cp",
	label: "Copy (virtual)",
	description: "Copy an entry in the virtual namespace. Original source files are NOT duplicated on disk.",
	guideline: "Use cp to present a file in multiple virtual locations without copying source data.",
	parameters: Type.Object({
		from: Type.String({ description: "Source virtual path" }),
		to: Type.String({ description: "Destination virtual path" }),
	}),
	async run(ws, params) {
		const from = String(params.from);
		const to = String(params.to);
		await ws.cp(from, to);
		return { text: `Copied ${from} -> ${to}`, details: { method: "cp", sources: [to], resultCount: 1 } };
	},
};

const mkdirSpec: ToolSpec<TSchema> = {
	name: "mkdir",
	label: "Mkdir (virtual)",
	description: "Create a virtual directory.",
	guideline: "Use mkdir to create a virtual folder before moving entries into it.",
	parameters: Type.Object({
		path: Type.String({ description: "Virtual directory path to create" }),
	}),
	async run(ws, params) {
		const path = String(params.path);
		await ws.mkdir(path);
		return { text: `Created ${path}`, details: { method: "mkdir", sources: [path], resultCount: 1 } };
	},
};

const rmdirSpec: ToolSpec<TSchema> = {
	name: "rmdir",
	label: "Rmdir (virtual)",
	description: "Remove a virtual directory (optionally recursive). Source files are NOT deleted.",
	guideline: "Use rmdir to remove a virtual folder. Pass recursive=true to remove children.",
	parameters: Type.Object({
		path: Type.String({ description: "Virtual directory path to remove" }),
		recursive: Type.Optional(Type.Boolean({ description: "Remove children recursively" })),
	}),
	async run(ws, params) {
		const path = String(params.path);
		await ws.rmdir(path, params.recursive === true);
		return { text: `Removed ${path}`, details: { method: "rmdir", sources: [path], resultCount: 1 } };
	},
};

const SPECS: ToolSpec<TSchema>[] = [
	grepSpec,
	findSpec,
	readSpec,
	lsSpec,
	statSpec,
	mvSpec,
	cpSpec,
	mkdirSpec,
	rmdirSpec,
];

function toResult(text: string, details: AgentdirToolDetails): AgentToolResult<AgentdirToolDetails> {
	return { content: [{ type: "text", text }], details };
}

/** A workspace handle or a (possibly async) provider resolved lazily at call time. */
export type WorkspaceProvider = Workspace | (() => Workspace | Promise<Workspace>);

async function resolveWorkspace(provider: WorkspaceProvider): Promise<Workspace> {
	return typeof provider === "function" ? provider() : provider;
}

/** Build agentdir tool definitions (extension surface via pi.registerTool). */
export function createAgentdirToolDefinitions(provider: WorkspaceProvider): ToolDefinition[] {
	return SPECS.map(
		(spec): ToolDefinition => ({
			name: spec.name,
			label: spec.label,
			description: spec.description,
			promptSnippet: spec.description,
			promptGuidelines: [spec.guideline],
			parameters: spec.parameters,
			async execute(_toolCallId, params) {
				const ws = await resolveWorkspace(provider);
				const { text, details } = await spec.run(ws, params as Record<string, unknown>);
				return toResult(text, details);
			},
		}),
	);
}

/** Build agentdir tools for the library Agent (options.tools). */
export function createAgentdirTools(provider: WorkspaceProvider): AgentTool[] {
	return SPECS.map(
		(spec): AgentTool => ({
			name: spec.name,
			label: spec.label,
			description: spec.description,
			parameters: spec.parameters,
			async execute(_toolCallId, params) {
				const ws = await resolveWorkspace(provider);
				const { text, details } = await spec.run(ws, params as Record<string, unknown>);
				return toResult(text, details);
			},
		}),
	);
}
