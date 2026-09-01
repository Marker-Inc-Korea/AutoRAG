import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { datasourceSourcePath } from "../../scope.ts";
import type {
	DatasourceDiagnostic,
	DatasourceDiagnosticCode,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SourceDescription,
} from "../../types.ts";
import {
	DiscrawlFtsMethod,
	DiscrawlHybridMethod,
	type DiscrawlSearchClient,
	DiscrawlSemanticMethod,
} from "./methods.ts";
import type {
	DiscrawlDoctorResult,
	DiscrawlEmbedResult,
	DiscrawlFailure,
	DiscrawlSearchMode,
	DiscrawlSyncResult,
} from "./types.ts";
import { DEFAULT_DISCRAWL_EMBEDDING_MODEL, DEFAULT_DISCRAWL_MODE, ENGLISH_ONLY_EMBEDDING_MODELS } from "./types.ts";

export interface DiscrawlSkillClient extends DiscrawlSearchClient {
	doctor(): Promise<DiscrawlDoctorResult>;
	sync(): Promise<DiscrawlSyncResult>;
	embed(limit?: number): Promise<DiscrawlEmbedResult>;
}

export interface DiscrawlSkillOptions {
	readonly client: DiscrawlSkillClient;
	readonly datasourceId?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly lastIndexedAt?: number;
	readonly embeddingModel?: string;
	readonly defaultMode?: DiscrawlSearchMode;
	/** Max messages embedded per index pass. Undefined drains the whole backlog. */
	readonly embedLimit?: number;
	readonly channelIds?: readonly string[];
	readonly channelNames?: readonly string[];
}

const DISCORD_DATASOURCE_ID = "discord";
const DISCORD_SKILL_TYPE = "discord-archive";
const DEFAULT_INSTANCE_ID = "default";
const DEFAULT_POLLING_INTERVAL_MS = 15 * 60 * 1000;
const DEFAULT_DISCORD_TAGS = ["discord", "chat", "pii"] as const;

/**
 * Discord datasource skill backed by the external `discrawl` CLI.
 *
 * Mirrors the katok model: the CLI owns the archive, the FTS5 index, and the
 * vector index; AutoRAG only spawns it and maps results. AutoRAG never calls
 * the Discord API itself and never reads the Discord Desktop cache directly.
 *
 * `index()` runs the CLI's two-phase pipeline — `sync` (incremental, cursor
 * based) then `embed` (drains the embedding queue) — so a poll costs work
 * proportional to what changed rather than to archive size.
 */
export class DiscrawlSkill implements DatasourceSkill {
	private readonly client: DiscrawlSkillClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly pollingIntervalMs: number;
	private readonly tags: readonly string[];
	private readonly embeddingModel: string;
	private readonly defaultMode: DiscrawlSearchMode;
	private readonly embedLimit: number | undefined;
	private lastIndexedAt: number | undefined;

	constructor(options: DiscrawlSkillOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.pollingIntervalMs = options.pollingIntervalMs ?? DEFAULT_POLLING_INTERVAL_MS;
		this.tags = options.tags ?? DEFAULT_DISCORD_TAGS;
		this.embeddingModel = options.embeddingModel ?? DEFAULT_DISCRAWL_EMBEDDING_MODEL;
		this.defaultMode = options.defaultMode ?? DEFAULT_DISCRAWL_MODE;
		this.embedLimit = options.embedLimit;
		this.lastIndexedAt = options.lastIndexedAt;
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: DISCORD_DATASOURCE_ID,
			id: DISCORD_DATASOURCE_ID,
			type: DISCORD_SKILL_TYPE,
			description: "Discord datasource via the external discrawl CLI",
			capabilities: ["chat", "external-cli", "polling", "fts5", "semantic", "hybrid", "incremental"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: DISCORD_DATASOURCE_ID,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: this.pollingIntervalMs > 0 ? "poll" : "none",
			...(this.pollingIntervalMs > 0 ? { intervalMs: this.pollingIntervalMs } : {}),
			lastIndexedAt: this.lastIndexedAt,
		};
	}

	async index(): Promise<DatasourceIndexResult> {
		const diagnostics: DatasourceDiagnostic[] = [];

		let doctor: DiscrawlDoctorResult;
		try {
			doctor = await this.client.doctor();
		} catch {
			return this.fail("datasource-unavailable", "discrawl doctor failed unexpectedly", diagnostics);
		}
		if (!doctor.ok) return this.fail(failureCode(doctor, "datasource-unavailable"), describe(doctor), diagnostics);
		const modelWarning = this.englishOnlyModelDiagnostic(doctor.data.embeddingModel ?? this.embeddingModel);
		if (modelWarning !== undefined) diagnostics.push(modelWarning);
		if (!doctor.data.databaseOk) {
			return this.fail("datasource-unavailable", "discrawl archive database is not ready", diagnostics);
		}

		let sync: DiscrawlSyncResult;
		try {
			sync = await this.client.sync();
		} catch {
			return this.fail("datasource-index-failed", "discrawl sync failed unexpectedly", diagnostics);
		}
		if (!sync.ok) return this.fail(failureCode(sync, "datasource-index-failed"), describe(sync), diagnostics);

		if (doctor.data.embeddingsOk) {
			const embed = await this.runEmbed();
			if (embed !== undefined) diagnostics.push(embed);
		} else {
			diagnostics.push({
				code: "datasource-index-failed",
				severity: "warning",
				message: "embeddings are not configured; semantic and hybrid retrieval will fall back to FTS only",
				instanceId: this.instanceId,
				source: DISCORD_DATASOURCE_ID,
			});
		}

		this.lastIndexedAt = Date.now();
		if (sync.data.messages === 0) {
			diagnostics.push({
				code: "datasource-empty",
				severity: "info",
				message: "discrawl sync found no new messages",
				instanceId: this.instanceId,
				source: DISCORD_DATASOURCE_ID,
			});
		}
		return {
			ok: true,
			instanceId: this.instanceId,
			skill: DISCORD_DATASOURCE_ID,
			chunkCount: sync.data.messages,
			indexedAt: this.lastIndexedAt,
			diagnostics,
		};
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		const options = { client: this.client, instanceId: this.instanceId, tags: this.tags };
		const hybrid = new DiscrawlHybridMethod(options);
		const fts = new DiscrawlFtsMethod(options);
		const semantic = new DiscrawlSemanticMethod(options);
		if (this.defaultMode === "fts") return [fts, semantic, hybrid];
		if (this.defaultMode === "semantic") return [semantic, fts, hybrid];
		return [hybrid, fts, semantic];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath(DISCORD_DATASOURCE_ID, instanceId),
			datasourceId: DISCORD_DATASOURCE_ID,
			skill: DISCORD_DATASOURCE_ID,
			instanceId,
			contentType: "chat",
			metadata: { datasourceId: DISCORD_DATASOURCE_ID, instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const cadence =
			this.pollingIntervalMs > 0
				? `roughly every ${Math.round(this.pollingIntervalMs / 60000)} minute(s) when auto-refresh runs`
				: "on manual refresh only";
		return {
			name: `datasource-${DISCORD_DATASOURCE_ID}`,
			description:
				"Search archived Discord messages across guilds, channels, threads, and DMs. Use for questions about Discord conversations, decisions made in channels, or who said what.",
			content: [
				`# Discord datasource (${DISCORD_SKILL_TYPE})`,
				"",
				"This skill searches a Discord archive maintained by the external `discrawl` CLI. AutoRAG never calls the Discord API directly and never reads the Discord Desktop cache itself.",
				"",
				"## When to use",
				"Use this skill when the question is about Discord conversations, channel discussions, or content shared inside a guild or DM.",
				"",
				"## Indexing",
				`Indexing is refreshed ${cadence} via \`discrawl sync\` and \`discrawl embed\` (incremental). You do not trigger indexing; just search.`,
				"",
				"## How to search",
				"Call `search_datasource_documents` with a natural-language `query` and `topK`.",
				"",
				"## Native CLI",
				"You can also invoke `discrawl` directly through `bash` when you need its full surface:",
				"- `discrawl --json search --mode fts <query> --limit <n>` — SQLite FTS5 lexical search",
				"- `discrawl --json search --mode semantic <query> --limit <n>` — vector search",
				"- `discrawl --json search --mode hybrid <query> --limit <n>` — combined retrieval",
				"- `discrawl doctor` — health and config check",
				"- `discrawl status --json` — archive freshness",
				"- `discrawl --help` — full command reference",
				"",
				"## Configuration",
				"discrawl reads its native config and archive (`~/Library/Application Support/discrawl/` on macOS) by default; AutoRAG never creates or manages them.",
				"Server-side connector overrides: `configPath` → `discrawl --config <file>`, `workspacePath` → child working directory, `guildId` → restrict sync/search to one guild, `source` → `wiretap` (local cache import) or `discord` (bot-token sync).",
				"",
				"## Retrieval quality",
				"Hybrid retrieval is the default. The underlying FTS index merges words across line breaks into a single token, so lexical-only search can miss terms that appear immediately after a newline; semantic recall covers that gap. Prefer natural-language queries over single exact keywords.",
				"",
				"## Output rules",
				`Source identifiers such as \`/${DISCORD_DATASOURCE_ID}/<instance>/chunks/<message-id>\` are stable Discord message ids and are traceable. Result metadata carries guild, channel, author, and timestamp; you may cite them when they help the user locate the message. Privacy is the operator's responsibility: run AutoRAG with a local LLM if results must not leave this machine.`,
			].join("\n"),
		};
	}

	private async runEmbed(): Promise<DatasourceDiagnostic | undefined> {
		let embed: DiscrawlEmbedResult;
		try {
			embed = await this.client.embed(this.embedLimit);
		} catch {
			return {
				code: "datasource-index-failed",
				severity: "warning",
				message: "discrawl embed failed unexpectedly; semantic retrieval may be stale",
				instanceId: this.instanceId,
				source: DISCORD_DATASOURCE_ID,
			};
		}
		if (!embed.ok) {
			return {
				code: "datasource-index-failed",
				severity: "warning",
				message: `discrawl embed failed (${embed.reason}); semantic retrieval may be stale`,
				instanceId: this.instanceId,
				source: DISCORD_DATASOURCE_ID,
			};
		}
		if (embed.data.failed > 0) {
			return {
				code: "datasource-index-failed",
				severity: "warning",
				message: `discrawl embed completed with ${embed.data.failed} failed message(s)`,
				instanceId: this.instanceId,
				source: DISCORD_DATASOURCE_ID,
			};
		}
		return undefined;
	}

	/**
	 * English-only embedders map non-English text into one narrow similarity
	 * band, so semantic search degrades to noise without ever erroring. Surfacing
	 * it as a diagnostic keeps that failure visible. See AutoRAG issue #1414.
	 */
	private englishOnlyModelDiagnostic(embeddingModel: string): DatasourceDiagnostic | undefined {
		if (!ENGLISH_ONLY_EMBEDDING_MODELS.has(embeddingModel.toLowerCase())) return undefined;
		return {
			code: "datasource-index-failed",
			severity: "warning",
			message: `embedding model "${embeddingModel}" is English-only; semantic search will be unreliable for non-English messages (prefer "${DEFAULT_DISCRAWL_EMBEDDING_MODEL}")`,
			instanceId: this.instanceId,
			source: DISCORD_DATASOURCE_ID,
		};
	}

	private fail(
		code: DatasourceDiagnosticCode,
		message: string,
		existing: readonly DatasourceDiagnostic[] = [],
	): DatasourceIndexResult {
		const diagnostic: DatasourceDiagnostic = {
			code,
			severity: code === "datasource-unavailable" || code === "datasource-empty" ? "warning" : "error",
			message,
			instanceId: this.instanceId,
			source: DISCORD_DATASOURCE_ID,
		};
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: DISCORD_DATASOURCE_ID,
			indexedAt: Date.now(),
			diagnostics: [...existing, diagnostic],
			error: message,
			code,
			message,
		};
	}
}

function failureCode(failure: DiscrawlFailure, fallback: DatasourceDiagnosticCode): DatasourceDiagnosticCode {
	switch (failure.reason) {
		case "binary-missing":
		case "spawn-error":
		case "timeout":
		case "aborted":
			return "datasource-unavailable";
		case "user-token-rejected":
			return "datasource-permission-denied";
		default:
			return fallback;
	}
}

function describe(failure: DiscrawlFailure): string {
	const detail = failure.stderr.length > 0 ? `: ${failure.stderr}` : "";
	return `${failure.reason}${detail}`;
}
