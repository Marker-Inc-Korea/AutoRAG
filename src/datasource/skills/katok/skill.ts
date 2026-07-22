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
import { KatokBm25Method, type KatokSearchClient, KatokSemanticMethod } from "./methods.ts";
import type { KatokDoctorResult, KatokIndexResult, KatokSyncResult } from "./types.ts";

export interface KatokSkillClient extends KatokSearchClient {
	doctor(): Promise<KatokDoctorResult>;
	sync(): Promise<KatokSyncResult>;
	index(): Promise<KatokIndexResult>;
}

export interface KatokSkillOptions {
	readonly client: KatokSkillClient;
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

export class KatokSkill implements DatasourceSkill {
	private readonly client: KatokSkillClient;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly pollingIntervalMs: number;
	private readonly tags: readonly string[];
	private lastIndexedAt: number | undefined;

	constructor(options: KatokSkillOptions) {
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
			description: "KakaoTalk datasource via the external katok CLI",
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
			if (!doctor.ok) return this.fail(katokFailureCode("datasource-unavailable", doctor.reason), doctor);
			const sync = await this.client.sync();
			if (!sync.ok) return this.fail(katokFailureCode("datasource-index-failed", sync.reason), sync);
			const indexResult = await this.client.index();
			if (!indexResult.ok)
				return this.fail(katokFailureCode("datasource-index-failed", indexResult.reason), indexResult);
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
		} catch {
			return this.fail("datasource-unavailable", {
				ok: false,
				reason: "spawn-error",
				stdout: "",
				stderr: "katok command failed; details suppressed for datasource privacy",
				code: null,
			});
		}
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [
			new KatokBm25Method({ client: this.client, instanceId: this.instanceId, tags: this.tags }),
			new KatokSemanticMethod({ client: this.client, instanceId: this.instanceId, tags: this.tags }),
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
		const instanceScopes = this.instances
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
				"This skill searches KakaoTalk chats that are indexed through the external `katok` CLI. AutoRAG never reads KakaoTalk databases directly.",
				"",
				"## When to use",
				"Use this skill when the question is about KakaoTalk conversations, chat participants, or content shared inside chats.",
				"",
				"## Indexing",
				`Indexing is server-managed and refreshed ${cadence}. You do not trigger indexing; just search.`,
				"",
				"## How to search",
				"Call `search_datasource_documents` with a natural-language `query`. Optionally pass `topK` and a narrowing `scope`. Available authorized scopes:",
				instanceScopes.length > 0 ? instanceScopes : "- (no authorized instances)",
				"",
				"`scope` can only narrow within already-authorized scopes; it can never widen access.",
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
			result.stderr.length > 0 ? `${result.reason}: ${sanitizeDiagnosticText(result.stderr)}` : result.reason;
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

function katokFailureCode(fallback: DatasourceDiagnosticCode, reason: string): DatasourceDiagnosticCode {
	return reason === "remote-embedding-rejected" ? "datasource-embedding-egress-rejected" : fallback;
}

function sanitizeDiagnosticText(value: string): string {
	if (value.includes("/") || value.includes("\\") || /[A-Za-z]:[\\/]/u.test(value)) {
		return "katok command failed; details suppressed for datasource privacy";
	}
	return value;
}
