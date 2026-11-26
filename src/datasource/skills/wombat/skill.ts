import { readFileSync } from "node:fs";
import { FallbackRetrievalMethod } from "../../../retrieval/fallback.ts";
import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { grepDirectoryWithRipgrep } from "../../grep.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";
import { datasourceSearchToolName } from "../../tool-naming.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceGrepHit,
	DatasourceGrepOptions,
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SearchFilterSpec,
	SourceDescription,
} from "../../types.ts";
import { validateWombatAgents, wombatMirrorDir } from "./agents.ts";
import { WombatClient, type WombatSyncClient } from "./client.ts";
import { WombatMethod } from "./methods.ts";
import { applyWombatQueryPrefix } from "./query-prefix.ts";
import type { WombatFailureReason, WombatOptions } from "./types.ts";

const TAGS = ["wombat", "coding-agent", "sessions", "pii"] as const;
const DEFAULT_POLLING_MS = 15 * 60 * 1000;

export interface WombatSkillOptions extends WombatOptions {
	readonly client?: WombatSyncClient;
	readonly datasourceId?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly connectorOptions?: WombatOptions;
}

/** An instance id becomes a path segment of every source, so it must be one safe segment. */
export function validateWombatInstanceId(instanceId: string): void {
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(instanceId)) {
		throw new Error("wombat instanceId must be a safe single path segment");
	}
}

/**
 * Coding-agent session history (Codex, Claude Code, OpenCode, pi, omp, and the other agents in `WOMBAT_AGENTS`) indexed by the external `wombat` CLI.
 * wombat parses each agent's own log format, keeps only the user and assistant conversation, and lets MinSync
 * index it incrementally; AutoRAG never reads those logs or the MinSync store itself.
 */
export class WombatSkill implements DatasourceSkill {
	private readonly client: WombatSyncClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly interval: number;
	private readonly tags: readonly string[];
	private readonly agents: readonly string[] | undefined;
	private readonly connectorOptions: WombatOptions;
	private lastIndexedAt: number | undefined;

	constructor(options: WombatSkillOptions = {}) {
		const connectorOptions = options.connectorOptions ?? options;
		this.instanceId = options.instanceId ?? "default";
		this.instances = options.instances?.length ? options.instances : [this.instanceId];
		validateWombatInstanceId(this.instanceId);
		for (const instance of this.instances) validateWombatInstanceId(instance);
		validateWombatAgents(connectorOptions);
		this.connectorOptions = connectorOptions;
		this.interval = options.pollingIntervalMs ?? DEFAULT_POLLING_MS;
		this.tags = options.tags ?? TAGS;
		this.agents = connectorOptions.agents?.length ? connectorOptions.agents : undefined;
		this.client = options.client ?? new WombatClient({ ...connectorOptions, instanceId: this.instanceId });
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: "wombat",
			id: "wombat",
			type: "coding-agent-sessions",
			description:
				"The user's past conversations with coding agents (Claude Code, Codex, OpenCode, pi, omp, Gemini, Kiro and more): what was asked, decided, or built in earlier sessions, in any project. Only the user and assistant text is indexed, not tool output or code diffs.",
			capabilities: ["coding-agent", "external-cli", "polling", "incremental", "bm25", "semantic", "hybrid"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: "wombat",
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: this.interval > 0 ? "poll" : "none",
			...(this.interval > 0 ? { intervalMs: this.interval } : {}),
			lastIndexedAt: this.lastIndexedAt,
		};
	}

	async index(options?: DatasourceIndexOptions): Promise<DatasourceIndexResult> {
		try {
			const doctor = await this.client.doctor(options?.signal);
			if (!doctor.ok) return this.fail(failureCode(doctor.reason), doctor.reason, doctor.stderr);
			const result = await this.client.sync(options?.signal);
			if (!result.ok) return this.fail(failureCode(result.reason), result.reason, result.stderr);
			const diagnostics: DatasourceDiagnostic[] = [];
			if (!result.data.indexOk || (result.data.failed ?? 0) > 0) {
				diagnostics.push({
					code: "datasource-index-partial",
					severity: "warning",
					message: `wombat could not parse or index ${result.data.failed ?? 0} session(s); the next refresh retries them`,
					instanceId: this.instanceId,
					source: "wombat",
				});
			}
			// The workspace exists once a sync has run; the instruction only changes how queries are embedded.
			applyWombatQueryPrefix(wombatMirrorDir(this.connectorOptions));
			const chunkCount = result.data.chunkCount ?? result.data.totalMessages;
			if (result.data.totalSessions === 0) {
				diagnostics.push({
					code: "datasource-empty",
					severity: "info",
					message: "wombat found no coding-agent sessions to index",
					instanceId: this.instanceId,
					source: "wombat",
				});
			}
			this.lastIndexedAt = Date.now();
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: "wombat",
				chunkCount,
				indexedAt: this.lastIndexedAt,
				diagnostics,
			};
		} catch {
			return this.fail("datasource-unavailable", "unexpected-error", "");
		}
	}

	/** Hybrid first; SPLADE keyword search when the dense side is unavailable (for example, no embedder yet). */
	retrievalMethods(): readonly RetrievalMethod[] {
		const hybrid = new WombatMethod(this.client, "hybrid", this.instanceId, this.tags, this.agents);
		const keyword = new WombatMethod(this.client, "keyword", this.instanceId, this.tags, this.agents);
		return [new FallbackRetrievalMethod(hybrid, [keyword])];
	}

	/**
	 * The one filter wombat offers: which coding agent's sessions to search. Ids are what the index holds now, narrowed to
	 * the connector's own list, so the model names an agent that exists. Empty when the index has no sessions or cannot be
	 * read (a missing CLI shows up through indexing, not as an empty filter).
	 */
	async searchFilters(): Promise<readonly SearchFilterSpec[]> {
		const listed = await this.client.indexedAgents?.();
		if (listed === undefined || !listed.ok) return [];
		const agents =
			this.agents === undefined ? listed.agents : listed.agents.filter((agent) => this.agents?.includes(agent.id));
		if (agents.length === 0) return [];
		const ids = agents.map((agent) => agent.id);
		return [
			{
				name: "agents",
				description:
					'Only search sessions of these coding agents (ids as listed). Use it when the question names one, for example "what did I ask Codex".',
				values: ids.map((id, index) => ({
					id,
					detail: `${agents[index]?.sessions} ${agents[index]?.sessions === 1 ? "session" : "sessions"}`,
				})),
			},
		];
	}

	/** Exact substring/regex grep over wombat's parsed markdown session mirror via ripgrep (zero model/MinSync calls). */
	async grep(pattern: string, options?: DatasourceGrepOptions): Promise<readonly DatasourceGrepHit[]> {
		if (pattern.length === 0) return [];
		const mirrorDir = wombatMirrorDir(this.connectorOptions);
		const matches = await grepDirectoryWithRipgrep({
			rootDir: mirrorDir,
			pattern,
			literal: options?.literal,
			ignoreCase: options?.ignoreCase,
			context: options?.context,
			limit: options?.limit,
			maxPerFile: options?.maxPerSource,
			skipDirs: [".minsync"],
			signal: options?.signal,
		});
		if (matches.length === 0) return [];

		const headerCache = new Map<string, ParsedWombatMirrorHeader>();
		const hits: DatasourceGrepHit[] = [];
		for (const match of matches) {
			let header = headerCache.get(match.absPath);
			if (header === undefined) {
				header = parseWombatMirrorHeader(match.absPath, match.relPath);
				headerCache.set(match.absPath, header);
			}
			if (this.agents !== undefined && !this.agents.includes(header.agent)) continue;
			const turn = locateTurnForLine(header.turnStartLines, match.lineNumber);
			const chunkId = `grep:${header.catalogKey}:L${match.lineNumber}`;
			const source = datasourceSourcePath("wombat", this.instanceId, chunkId);
			if (!matchesDatasourceScope(source, options?.scope)) continue;
			const labelParts = [
				header.catalogKey,
				turn !== undefined ? `turn ${turn}` : undefined,
				header.startedAt?.slice(0, 10),
				header.cwd,
			].filter((part): part is string => part !== undefined && part.length > 0);
			const label = labelParts.join(" | ");
			hits.push({
				source,
				content: `[${label}]\n${match.snippet}`,
				lineNumber: match.lineNumber,
				lineEnd: match.lineEnd,
				label,
				metadata: {
					datasourceId: "wombat",
					instanceId: this.instanceId,
					agent: header.agent,
					sessionId: header.sessionId,
					sessionKey: header.catalogKey,
					...(header.cwd !== undefined ? { cwd: header.cwd } : {}),
					...(header.sourcePath !== undefined ? { sessionPath: header.sourcePath } : {}),
					...(turn !== undefined ? { startTurn: turn, endTurn: turn } : {}),
				},
			});
		}
		return hits;
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath("wombat", instanceId),
			datasourceId: "wombat",
			skill: "wombat",
			instanceId,
			contentType: "coding-agent-session",
			metadata: { datasourceId: "wombat", instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const scopes = this.instances.map((id) => `- \`${datasourceSourcePath("wombat", id)}\``).join("\n");
		return {
			name: "datasource-wombat",
			description:
				"Search past coding-agent conversations (Codex, Claude Code, OpenCode, pi, omp, Gemini, Kiro, and more): what was asked, decided, or built. Each hit names its session. Load this skill to open that session's full conversation or its original log with the tool calls that the search does not index.",
			content: [
				"# wombat coding-agent session datasource",
				"",
				"Use this skill to recall what the user and a coding agent discussed in earlier sessions. Only the user and assistant conversation is indexed. Thinking, tool calls and outputs, and file diffs are not, so search for what was said, not for code that was run.",
				"",
				"## Indexing",
				"Indexing is server-managed and incremental; do not trigger sync yourself.",
				"",
				"## Search",
				`Call the dedicated \`${datasourceSearchToolName("wombat")}\` tool with a natural-language query. Available scopes:`,
				scopes,
				"",
				"The selectable retrieval modes are keyword (SPLADE), semantic, and hybrid. Scope arguments can only narrow trusted access. Source identifiers remain opaque and hierarchical.",
				"Every result starts with a label line, `[<sessionKey> | turns A-B | date | title | project]`, followed by the snippet. `sessionKey` is `<agent>:<session id>`.",
				"",
				"## Native CLI",
				"The external `wombat` CLI owns the mirror, the index and the session catalog. To read more than a snippet, call it through `bash` with the `sessionKey` from the result's label:",
				"- `wombat show <sessionKey> [--turn N]`: the full cleaned conversation, or one turn.",
				"- `wombat trace <sessionKey> --turn N --context K`: the original session log around that turn, including the tool calls and outputs that the index leaves out.",
				"- `wombat grep [--literal] [--ignore-case] [--agent <agent>] <pattern> --json`: exact ripgrep match over parsed session mirrors without invoking MinSync.",
				"- `wombat chunk context <chunk id> --json`: the turns before and after a result.",
				"- `wombat sessions --agent <agent>`: list sessions.",
				'Or call the `grep` tool with `path="/wombat"` for exact-match lookup registered directly as `[eN]` evidence.',
			].join("\n"),
		};
	}

	private fail(code: DatasourceDiagnosticCode, reason: string, stderr: string): DatasourceIndexResult {
		const detail = stderr.trim().length > 0 ? `${reason}: ${stderr.trim()}` : reason;
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: "wombat",
			indexedAt: Date.now(),
			diagnostics: [
				{
					code,
					severity: code === "datasource-unavailable" ? "warning" : "error",
					message: `wombat ${detail}`,
					instanceId: this.instanceId,
					source: "wombat",
				},
			],
			error: detail,
			code,
			message: detail,
		};
	}
}

function failureCode(reason: WombatFailureReason): DatasourceDiagnosticCode {
	if (
		reason === "binary-missing" ||
		reason === "spawn-error" ||
		reason === "timeout" ||
		reason === "minsync-unavailable"
	)
		return "datasource-unavailable";
	if (reason === "remote-embedding-rejected") return "datasource-embedding-egress-rejected";
	return "datasource-index-failed";
}

interface ParsedWombatMirrorHeader {
	readonly agent: string;
	readonly sessionId: string;
	readonly catalogKey: string;
	readonly sourcePath?: string;
	readonly cwd?: string;
	readonly startedAt?: string;
	readonly turnStartLines: readonly { readonly line: number; readonly turn: number }[];
}

function parseWombatMirrorHeader(absPath: string, relPath: string): ParsedWombatMirrorHeader {
	const fallbackAgent = relPath.split("/")[0] || "unknown";
	let agent = fallbackAgent;
	let sessionId = relPath;
	let sourcePath: string | undefined;
	let cwd: string | undefined;
	let startedAt: string | undefined;
	const turnStartLines: { line: number; turn: number }[] = [];
	try {
		const lines = readFileSync(absPath, "utf8").split(/\r?\n/u);
		for (let i = 0; i < lines.length; i += 1) {
			const line = lines[i] ?? "";
			if (i < 12) {
				if (line.startsWith("- agent: ")) agent = line.slice("- agent: ".length).trim() || agent;
				else if (line.startsWith("- session_id: "))
					sessionId = line.slice("- session_id: ".length).trim() || sessionId;
				else if (line.startsWith("- source_path: "))
					sourcePath = line.slice("- source_path: ".length).trim() || undefined;
				else if (line.startsWith("- cwd: ")) cwd = line.slice("- cwd: ".length).trim() || undefined;
				else if (line.startsWith("- started_at: "))
					startedAt = line.slice("- started_at: ".length).trim() || undefined;
			}
			const match = /^## Turn (\d+)\b/u.exec(line);
			if (match?.[1] !== undefined) {
				turnStartLines.push({ line: i + 1, turn: Number(match[1]) });
			}
		}
	} catch {
		// Fall back to relative-path metadata if unreadable.
	}
	return {
		agent,
		sessionId,
		catalogKey: `${agent}:${sessionId}`,
		...(sourcePath !== undefined ? { sourcePath } : {}),
		...(cwd !== undefined ? { cwd } : {}),
		...(startedAt !== undefined ? { startedAt } : {}),
		turnStartLines,
	};
}

function locateTurnForLine(
	turnStartLines: readonly { readonly line: number; readonly turn: number }[],
	lineNumber: number,
): number | undefined {
	let best: number | undefined;
	for (const entry of turnStartLines) {
		if (entry.line <= lineNumber) best = entry.turn;
		else break;
	}
	return best ?? turnStartLines[0]?.turn;
}
