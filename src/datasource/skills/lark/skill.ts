import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import type { CrawlerFailure } from "../../crawler-types.ts";
import { datasourceCliError } from "../../errors.ts";
import { matchesDatasourceScope, normalizeVirtualPath } from "../../scope.ts";
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
import { isSafeId, LarkClient, type LarkClientOptions } from "./client.ts";

const CHAT_TAG = "lark:chat";
const DOCS_TAG = "lark:docs";
const DEFAULT_TAGS = [CHAT_TAG, DOCS_TAG] as const;
const REMOTE_NOTE = "remote-search datasource: no local mirror";

export interface LarkSkillOptions extends LarkClientOptions {
	readonly client?: LarkClient;
	readonly datasourceId?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly tags?: readonly string[];
	readonly embedderBaseUrl?: string;
}

type Surface = "messages" | "docs";

export class LarkSkill implements DatasourceSkill {
	private readonly client: LarkClient;
	private readonly datasourceId: string;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly tags: readonly string[];
	private readonly chatIds: readonly string[];
	private readonly embedderBaseUrl: string | undefined;
	private lastIndexedAt: number | undefined;
	private lastError: string | undefined;

	constructor(options: LarkSkillOptions = {}) {
		this.datasourceId = options.datasourceId ?? "lark";
		this.instanceId = isSafeId(options.instanceId ?? "default") ? (options.instanceId ?? "default") : "default";
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.tags = options.tags ?? DEFAULT_TAGS;
		this.chatIds = options.chatIds ?? [];
		this.embedderBaseUrl = options.embedderBaseUrl;
		this.client =
			options.client ??
			new LarkClient({
				...(options.binaryPath !== undefined ? { binaryPath: options.binaryPath } : {}),
				...(options.env !== undefined ? { env: options.env } : {}),
				...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
				...(options.chatIds !== undefined ? { chatIds: options.chatIds } : {}),
				...(options.sleep !== undefined ? { sleep: options.sleep } : {}),
			});
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: this.datasourceId,
			id: this.datasourceId,
			type: "lark-remote",
			description: "Lark/Feishu remote search via lark-cli. Credentials stay in the CLI keychain.",
			capabilities: ["external-cli", "remote-search", "scoped"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: this.datasourceId,
			instanceId: this.instanceId,
			instances: this.instances,
		};
	}

	polling(): PollingMetadata {
		return {
			mode: "none",
			...(this.lastIndexedAt !== undefined ? { lastIndexedAt: this.lastIndexedAt } : {}),
			...(this.lastError !== undefined ? { lastError: this.lastError } : {}),
		};
	}

	async index(): Promise<DatasourceIndexResult> {
		try {
			if (isRemoteEmbedder(this.embedderBaseUrl)) {
				return this.fail("datasource-embedding-egress-rejected", "remote embedding egress is not allowed for lark");
			}
			const diagnostics: DatasourceDiagnostic[] = [this.note(REMOTE_NOTE, "datasource-empty", "info")];
			const version = await this.client.probeVersion();
			if (!version.ok) {
				const message = probeMessage("--version", version);
				this.lastError = message;
				diagnostics.push(this.note(message, "datasource-unavailable", "warning"));
				return this.ok(diagnostics);
			}
			const auth = await this.client.probeAuth();
			if (!auth.ok) {
				const message = probeMessage("auth status", auth);
				this.lastError = message;
				diagnostics.push(this.note(message, "datasource-auth-error", "warning"));
				return this.ok(diagnostics);
			}
			this.lastError = undefined;
			this.lastIndexedAt = Date.now();
			return this.ok(diagnostics);
		} catch (error) {
			const message = error instanceof Error ? error.message : "lark index failed";
			return this.fail("datasource-unavailable", message);
		}
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [new LarkRemoteMethod(this, "messages"), new LarkRemoteMethod(this, "docs")];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: normalizeVirtualPath(`/${this.datasourceId}/${instanceId}`),
			datasourceId: this.datasourceId,
			skill: this.datasourceId,
			instanceId,
			contentType: "lark",
			metadata: { datasourceId: this.datasourceId, instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const scopes = this.instances
			.map((instanceId) => `- \`${normalizeVirtualPath(`/${this.datasourceId}/${instanceId}`)}\``)
			.join("\n");
		const chats =
			this.chatIds.length === 0
				? "Message search covers every chat the logged-in identity can already see."
				: `Message search is limited to configured chat ids: ${this.chatIds.join(", ")}.`;
		return {
			name: `datasource-${this.datasourceId}`,
			description:
				"Search the signed-in Lark or Feishu tenant for messages and cloud documents. The tenant keeps the index; AutoRAG does not copy it.",
			content: [
				"# Lark / Feishu (lark-cli remote search)",
				"",
				"Search live Lark/Feishu messages and cloud documents through the official `lark-cli`. AutoRAG does not open the desktop client's private databases, does not keep a local archive, and never sees app secrets or user tokens. Those stay in the CLI keychain.",
				"",
				"## When to use",
				"Questions about Lark/Feishu chats or docs (docx, wiki, drive, sheets, bitable) the signed-in user can already open.",
				"",
				"## How to search",
				`Call the dedicated \`${datasourceSearchToolName(this.datasourceId)}\` tool with a query and optional narrowing scope. Available authorized scopes:`,
				scopes,
				"",
				"`scope` can only narrow within already-authorized scopes; it can never widen access. A scope under `messages` searches chats. A scope under `docs` searches documents.",
				"",
				chats,
				"",
				"Server ranking is not BM25, vector, or hybrid. Coverage of old messages, retention, and chat types is whatever the tenant indexed for this token, and that coverage is not guaranteed. A hit is a live pointer: later edits or recalls change the text. Document search rejects a query longer than 30 Unicode characters.",
				"",
				"## Read a hit",
				"Search snippets are not the whole record. Read them with the CLI, not with `bash` on the `/lark/...` identity:",
				"- `lark-cli im +messages-mget --message-ids <message_id> --format json`",
				"- `lark-cli docs +fetch --doc <token> --doc-format markdown`",
				"",
				"## Native CLI",
				"The external `lark-cli` owns login and credentials. When the dedicated tool cannot express a filter, call it through `bash`:",
				'- `lark-cli im +messages-search --query "<query>" --format json`',
				'- `lark-cli drive +search --query "<query>" --format json`',
				"Login is `lark-cli auth login` with explicit scopes (`search:message` for chats, `search:docs:read` for documents), then `lark-cli auth status`. `auth login --recommend` does not grant those scopes by itself.",
				`Never pass datasource virtual paths (\`/${this.datasourceId}/...\`) to bash; they are not OS paths.`,
			].join("\n"),
		};
	}

	async search(surface: Surface, query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		if (!surfaceSelected(options.scope, this.datasourceId, this.instanceId, surface)) return [];
		const result =
			surface === "messages"
				? await this.client.searchMessages(trimmed, options.topK)
				: await this.client.searchDocs(trimmed, options.topK);
		if (!result.ok) throw datasourceCliError(this.datasourceId, "search", result);
		const hits: RetrievalResult[] = [];
		for (const hit of result.hits) {
			if (!isSafeId(hit.id)) continue;
			const source = normalizeVirtualPath(`/${this.datasourceId}/${this.instanceId}/${surface}/${hit.id}`);
			if (source.includes("#")) continue;
			hits.push({
				id: hit.id,
				content: hit.content,
				source,
				score: hit.score,
				metadata: {
					...(hit.metadata ?? {}),
					datasourceId: this.datasourceId,
					instanceId: this.instanceId,
					surface,
				},
			});
		}
		return hits;
	}

	methodTags(surface: Surface): readonly string[] {
		if (this.tags === DEFAULT_TAGS) return [surface === "messages" ? CHAT_TAG : DOCS_TAG];
		return this.tags;
	}

	private ok(diagnostics: readonly DatasourceDiagnostic[]): DatasourceIndexResult {
		return {
			ok: true,
			instanceId: this.instanceId,
			skill: this.datasourceId,
			chunkCount: 0,
			indexedAt: this.lastIndexedAt ?? Date.now(),
			diagnostics,
		};
	}

	private fail(code: DatasourceDiagnosticCode, message: string): DatasourceIndexResult {
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: this.datasourceId,
			indexedAt: Date.now(),
			diagnostics: [this.note(message, code, "error")],
			error: message,
			code,
			message,
		};
	}

	private note(
		message: string,
		code: DatasourceDiagnosticCode,
		severity: DatasourceDiagnostic["severity"],
	): DatasourceDiagnostic {
		return { code, severity, message, instanceId: this.instanceId, source: this.datasourceId };
	}
}

class LarkRemoteMethod implements RetrievalMethod {
	private readonly skill: LarkSkill;
	private readonly surface: Surface;

	constructor(skill: LarkSkill, surface: Surface) {
		this.skill = skill;
		this.surface = surface;
	}

	describe(): RetrievalMethodDescriptor {
		const descriptor = this.skill.describe();
		return {
			name: `${descriptor.datasourceId ?? "lark"}-${this.surface}`,
			type: "remote",
			description:
				this.surface === "messages"
					? "Server-side Lark/Feishu message search via lark-cli"
					: "Server-side Lark/Feishu document search via lark-cli",
			status: "active",
			capabilities: ["remote-search", "scoped", "external-cli", "path-opaque-sources"],
			datasourceId: descriptor.datasourceId,
			tags: this.skill.methodTags(this.surface),
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return this.skill.search(this.surface, query, options);
	}
}

function surfaceSelected(
	scope: string | undefined,
	datasourceId: string,
	instanceId: string,
	surface: Surface,
): boolean {
	if (scope === undefined || scope.trim().length === 0) return true;
	const root = normalizeVirtualPath(`/${datasourceId}/${instanceId}/${surface}`);
	const normalized = normalizeVirtualPath(scope);
	if (normalized === root || normalized.startsWith(`${root}/`)) return true;
	return matchesDatasourceScope(root, normalized);
}

function probeMessage(command: string, failure: CrawlerFailure): string {
	const stderr = failure.stderr.trim();
	const exit = failure.code === null || failure.code === undefined ? "no exit code" : `exit code ${failure.code}`;
	return `lark-cli ${command} failed (${failure.reason}, ${exit})${stderr.length > 0 ? `: ${stderr}` : ""}`;
}

function isRemoteEmbedder(value: string | undefined): boolean {
	if (value === undefined) return false;
	const trimmed = value.trim().toLowerCase();
	return trimmed.startsWith("https://") || trimmed.startsWith("http://");
}
