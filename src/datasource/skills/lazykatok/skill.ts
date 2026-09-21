import { describeRetrievalError } from "../../../retrieval/skip.ts";
import type { RetrievalMethod } from "../../../retrieval/types.ts";
import { datasourceSourcePath } from "../../scope.ts";
import { datasourceSearchToolName } from "../../tool-naming.ts";
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
import { LazykatokBm25Method, type LazykatokSearchClient, LazykatokSemanticMethod } from "./methods.ts";
import type { LazykatokDoctorResult, LazykatokIndexResult, LazykatokSyncResult } from "./types.ts";

export interface LazykatokSkillClient extends LazykatokSearchClient {
	doctor(): Promise<LazykatokDoctorResult>;
	sync(): Promise<LazykatokSyncResult>;
	index(): Promise<LazykatokIndexResult>;
}

export interface LazykatokSkillOptions {
	readonly client: LazykatokSkillClient;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly pollingIntervalMs?: number;
	readonly tags?: readonly string[];
	readonly lastIndexedAt?: number;
}

const KAKAO_DATASOURCE_ID = "kakao";
const KAKAO_SKILL_TYPE = "kakaotalk";
const DEFAULT_INSTANCE_ID = "default";
const DEFAULT_POLLING_INTERVAL_MS = 15 * 60 * 1000;
const DEFAULT_KAKAO_TAGS = ["kakaotalk", "personal", "pii"] as const;

export class LazykatokSkill implements DatasourceSkill {
	private readonly client: LazykatokSkillClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly pollingIntervalMs: number;
	private readonly tags: readonly string[];
	private lastIndexedAt: number | undefined;

	constructor(options: LazykatokSkillOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.pollingIntervalMs = options.pollingIntervalMs ?? DEFAULT_POLLING_INTERVAL_MS;
		this.tags = options.tags ?? DEFAULT_KAKAO_TAGS;
		this.lastIndexedAt = options.lastIndexedAt;
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: KAKAO_DATASOURCE_ID,
			id: KAKAO_DATASOURCE_ID,
			type: KAKAO_SKILL_TYPE,
			description: "KakaoTalk datasource via the external lazykatok CLI",
			capabilities: ["chat", "external-cli", "polling", "bm25", "semantic"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: KAKAO_DATASOURCE_ID,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: "poll",
			intervalMs: this.pollingIntervalMs,
			lastIndexedAt: this.lastIndexedAt,
		};
	}

	async index(): Promise<DatasourceIndexResult> {
		try {
			const doctor = await this.client.doctor();
			if (!doctor.ok) return this.fail("datasource-unavailable", doctor);
			const sync = await this.client.sync();
			if (!sync.ok) return this.fail("datasource-index-failed", sync);
			const indexResult = await this.client.index();
			if (!indexResult.ok) return this.fail("datasource-index-failed", indexResult);
			this.lastIndexedAt = Date.now();
			const chunkCount =
				"data" in indexResult &&
				typeof indexResult.data === "object" &&
				indexResult.data !== null &&
				"chunkCount" in indexResult.data &&
				typeof indexResult.data.chunkCount === "number"
					? indexResult.data.chunkCount
					: 0;
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: KAKAO_DATASOURCE_ID,
				chunkCount,
				indexedAt: this.lastIndexedAt,
				diagnostics: [],
			};
		} catch (error) {
			// The unexpected failure reaches the operator verbatim: no placeholder,
			// no swallowed message.
			return this.fail("datasource-unavailable", {
				ok: false,
				reason: "spawn-error",
				stdout: "",
				stderr: describeRetrievalError(error),
				code: null,
			});
		}
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [
			new LazykatokBm25Method({ client: this.client, instanceId: this.instanceId, tags: this.tags }),
			new LazykatokSemanticMethod({ client: this.client, instanceId: this.instanceId, tags: this.tags }),
		];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath(KAKAO_DATASOURCE_ID, instanceId),
			datasourceId: KAKAO_DATASOURCE_ID,
			skill: KAKAO_DATASOURCE_ID,
			instanceId,
			contentType: "chat",
			metadata: {
				datasourceId: KAKAO_DATASOURCE_ID,
				instanceId,
				tags: this.tags,
			},
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const instanceSources = this.instances
			.map((instanceId) => `- \`${datasourceSourcePath(KAKAO_DATASOURCE_ID, instanceId)}\``)
			.join("\n");
		const cadence =
			this.pollingIntervalMs > 0
				? `roughly every ${Math.round(this.pollingIntervalMs / 60000)} minute(s) when auto-refresh runs`
				: "on manual refresh only";
		return {
			name: `datasource-${KAKAO_DATASOURCE_ID}`,
			description:
				"Search indexed KakaoTalk chat history (messages, senders, room context). Use for questions about KakaoTalk conversations, decisions made in chats, or who said what.",
			content: [
				`# KakaoTalk datasource (${KAKAO_SKILL_TYPE})`,
				"",
				"This skill searches KakaoTalk chats that are indexed through the external `lazykatok` CLI. AutoRAG never reads KakaoTalk databases directly.",
				"",
				"## When to use",
				"Use this skill when the question is about KakaoTalk conversations, chat participants, or content shared inside chats.",
				"",
				"## Indexing",
				`Indexing is server-managed and refreshed ${cadence}. You do not trigger indexing; just search.`,
				"",
				"## How to search",
				`Call the dedicated \`${datasourceSearchToolName(KAKAO_DATASOURCE_ID)}\` tool with a natural-language \`query\` and \`topK\`. This datasource does not support per-source scope narrowing. Authorized datasource:`,
				instanceSources.length > 0 ? instanceSources : "- (no configured instances)",
				"",
				"Access is controlled by the trusted datasource tag; chat/channel filtering is owned by lazykatok.",
				"",
				"## Native CLI",
				"The external `lazykatok` CLI owns the archive, index, and credentials. When the dedicated tool cannot express what you need, call lazykatok directly through `bash`:",
				'- `lazykatok search bm25 "<query>" --json --limit 20` — lexical search (`semantic` mode also available)',
				"- `lazykatok chunk get <chunkId> --json` — fetch one chunk by id (chunk ids appear in result metadata)",
				"- `lazykatok chunk context <chunkId> --json` — surrounding messages of a chunk",
				"Never pass datasource virtual paths (`/kakao/...`) to bash; they are not OS paths.",
				"",
				"## Output rules",
				"Datasource source identifiers such as `/kakao/<instance>/chunks/<id>` are internal and opaque. Never put them, real file paths, account IDs, or phone numbers in the visible answer.",
			].join("\n"),
		};
	}

	private fail(
		code: DatasourceDiagnosticCode,
		result: { ok: false; reason: string; stdout?: string; stderr: string; code: number | null },
	): DatasourceIndexResult {
		const message =
			result.stderr.length > 0 ? `${result.reason}: ${result.stderr.trim().slice(0, 4000)}` : result.reason;
		const diagnostic: DatasourceDiagnostic = {
			code,
			severity: code === "datasource-unavailable" ? "warning" : "error",
			message,
			instanceId: this.instanceId,
			source: KAKAO_DATASOURCE_ID,
		};
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: KAKAO_DATASOURCE_ID,
			indexedAt: Date.now(),
			diagnostics: [diagnostic],
			error: result.reason,
			code,
			message,
		};
	}
}

// The lazykatok CLI's stderr reaches the operator verbatim, paths included — the
// client bounds length; nothing is replaced with a placeholder.
