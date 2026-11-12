import { FallbackRetrievalMethod } from "../../../retrieval/fallback.ts";
import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { datasourceSourcePath } from "../../scope.ts";
import { datasourceSearchToolName } from "../../tool-naming.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "../../types.ts";
import { WombatClient, type WombatSyncClient } from "./client.ts";
import { WombatMethod } from "./methods.ts";
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
 * Coding-agent session history (Codex, Claude Code, OpenCode, pi, omp) indexed by the external `wombat` CLI.
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
	private lastIndexedAt: number | undefined;

	constructor(options: WombatSkillOptions = {}) {
		const connectorOptions = options.connectorOptions ?? options;
		this.instanceId = options.instanceId ?? "default";
		this.instances = options.instances?.length ? options.instances : [this.instanceId];
		validateWombatInstanceId(this.instanceId);
		for (const instance of this.instances) validateWombatInstanceId(instance);
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
			description: "Coding-agent session history datasource via the external wombat CLI",
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
			this.lastIndexedAt = Date.now();
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: "wombat",
				chunkCount: result.data.chunkCount ?? result.data.totalMessages,
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
				"Search past coding-agent conversations (Codex, Claude Code, OpenCode, pi, omp): what was asked, decided, or built. Each hit names its session. Load this skill to open that session's full conversation or its original log with the tool calls that the search does not index.",
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
				"- `wombat chunk context <chunk id> --json`: the turns before and after a result.",
				"- `wombat sessions --agent <agent>`: list sessions.",
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
	if (reason === "binary-missing" || reason === "spawn-error" || reason === "timeout") return "datasource-unavailable";
	if (reason === "remote-embedding-rejected") return "datasource-embedding-egress-rejected";
	return "datasource-index-failed";
}
