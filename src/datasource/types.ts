/**
 * Datasource core contract.
 *
 * A *datasource* is an external, server-bound source of retrieval evidence
 * (e.g. a KakaoTalk export reached through the external `katok` CLI). The
 * datasource layer sits *on top of* the existing retrieval pipeline: it owns
 * access gating and slash-hierarchical source naming, while actual retrieval
 * still flows through {@link RetrievalMethod} instances returned by a skill.
 *
 * Security invariants (enforced by {@link DatasourceAccessContext}):
 *  - Access is **default-deny**: when no trusted allow-tags are configured,
 *    every datasource descriptor and source is denied.
 *  - Deny is always an explicit `false` boolean / predicate return — never
 *    `undefined`-as-deny.
 *  - Model/tool arguments never grant access. Only the trusted server-supplied
 *    {@link DatasourceAccessContext} can authorize a datasource.
 *  - Datasource sources are slash-hierarchical opaque paths such as
 *    `/kakao/<instance-id>/chunks/<chunk-id>`; `#` fragments are rejected.
 */

import type { RetrievalMethod } from "../retrieval/types.ts";

export type { RetrievalMethod } from "../retrieval/types.ts";

/**
 * Descriptor for a datasource skill (e.g. KakaoTalk via `katok`).
 *
 * Structurally compatible with {@link RetrievalMethodDescriptor} so that
 * retrieval method descriptors can be gated by the same access context.
 */
export interface DatasourceSkillDescriptor {
	/** Stable skill name, e.g. `"kakao"`. */
	readonly name: string;
	/** Compatibility alias for public API examples; equal to `name` when present. */
	readonly id?: string;
	/** Skill kind, e.g. `"kakao"` or `"chat-export"`. */
	readonly type: string;
	readonly description: string;
	/** Capability tags such as `"chat"`, `"external-cli"`, `"polling"`. */
	readonly capabilities: readonly string[];
	/**
	 * Authorization tags. A descriptor is accessible only when at least one tag
	 * intersects the trusted allow-tags on the access context.
	 */
	readonly tags: readonly string[];
	readonly status: "active" | "stub";
	/** True when the skill shells out to an external CLI (e.g. `katok`). */
	readonly requiresExternalCli?: boolean;
	/**
	 * Set when this descriptor describes a datasource-backed surface.
	 * Non-datasource descriptors (e.g. plain `posix` retrieval methods) leave
	 * this undefined and are passed through by the access context.
	 */
	readonly datasourceId?: string;
	/** Optional primary instance id for single-instance skills. */
	readonly instanceId?: string;
	/** Optional list of instance ids this skill reports. */
	readonly instances?: readonly string[];
}

/**
 * Minimal structural shape required to gate a descriptor by access context.
 * Both {@link DatasourceSkillDescriptor} and `RetrievalMethodDescriptor`
 * satisfy this interface structurally.
 */
export interface DatasourceAccessible {
	readonly datasourceId?: string;
	readonly tags?: readonly string[];
}

/**
 * A pluggable datasource skill. Skills are server-bound: they are constructed
 * with trusted configuration and never derive authority from model/tool args.
 * Methods never throw — failures surface as {@link DatasourceDiagnostic}
 * entries on a {@link DatasourceIndexResult} fail variant.
 */
export interface DatasourceSkill {
	describe(): DatasourceSkillDescriptor;
	/** Current polling/indexing metadata for this skill. */
	polling(): PollingMetadata;
	/**
	 * Index (or re-index) the skill's instances. Returns an ok/fail union;
	 * never throws. Remote embedding egress MUST be rejected here and reported
	 * via a `datasource-embedding-egress-rejected` diagnostic.
	 */
	index(): Promise<DatasourceIndexResult>;
	/** Retrieval methods exposed by this skill, fed into the shared pipeline. */
	retrievalMethods(): readonly RetrievalMethod[];
	/** Opaque source descriptions for the skill's current evidence set. */
	describeSources(): readonly SourceDescription[];
	/**
	 * Progressive-disclosure agent-skill manifest (Pi agent-skill layer). The
	 * `name`/`description` appear in the system prompt; `content` is loaded on
	 * demand via the `load_datasource_skill` tool.
	 */
	skillManifest(): DatasourceSkillManifest;
}

/**
 * Progressive-disclosure agent-skill manifest for a datasource skill.
 *
 * Mirrors the Pi agent-skill layer (`Skill` from `@earendil-works/pi-agent-core`):
 * `name` + `description` are injected into the system prompt for progressive
 * disclosure, and `content` is the full skill instruction body loaded on demand
 * via the model's `load_datasource_skill` tool call. `content` MUST stay
 * path/PII opaque — reference only slash-hierarchical opaque scopes.
 */
export interface DatasourceSkillManifest {
	/** Stable, model-visible skill name used for lookup and the skills listing. */
	readonly name: string;
	/** Short "when to use" description for the system-prompt skills block. */
	readonly description: string;
	/** Full skill instructions (SKILL.md-style), loaded on demand. */
	readonly content: string;
}

/**
 * A configured, running instance of a datasource skill (e.g. one KakaoTalk
 * account reached through `katok`).
 */
export interface DatasourceInstance {
	/** Stable instance id, unique within a skill. */
	readonly id: string;
	readonly skill: DatasourceSkill;
	readonly descriptor: DatasourceSkillDescriptor;
	/** Slash-hierarchical opaque root, e.g. `/kakao/<instance-id>`. */
	readonly sourcePath: string;
	readonly polling?: PollingMetadata;
}

/** Polling / indexing metadata for a datasource skill or instance. */
export interface PollingMetadata {
	readonly mode: "none" | "poll" | "cron";
	readonly intervalMs?: number;
	readonly cronExpr?: string;
	/** Epoch milliseconds of the last successful index. */
	readonly lastIndexedAt?: number;
	/** Epoch milliseconds of the last poll attempt. */
	readonly lastPolledAt?: number;
	readonly lastError?: string;
}

/** Successful indexing outcome. */
export interface DatasourceIndexOk {
	readonly ok: true;
	readonly instanceId: string;
	readonly skill: string;
	readonly chunkCount: number;
	readonly indexedAt: number;
	readonly diagnostics: readonly DatasourceDiagnostic[];
}

/** Failed indexing outcome. Never thrown; returned with diagnostics. */
export interface DatasourceIndexFail {
	readonly ok: false;
	readonly instanceId: string;
	readonly skill: string;
	readonly indexedAt: number;
	readonly diagnostics: readonly DatasourceDiagnostic[];
	readonly error?: string;
	readonly code?: DatasourceDiagnosticCode;
	readonly message?: string;
}

/**
 * Index result — an ok/fail discriminated union. Skills MUST NOT throw from
 * `index()`; they return a fail variant instead.
 */
export type DatasourceIndexResult = DatasourceIndexOk | DatasourceIndexFail;

export type DatasourceDiagnosticCode =
	| "datasource-unavailable"
	| "datasource-cli-error"
	| "datasource-empty"
	| "datasource-rate-limited"
	| "datasource-auth-error"
	| "datasource-embedding-egress-rejected"
	| "datasource-index-failed"
	| "datasource-permission-denied"
	| "datasource-remote-embedding-rejected";

/** Path-opaque diagnostic emitted by the datasource layer. */
export interface DatasourceDiagnostic {
	readonly code: DatasourceDiagnosticCode;
	readonly severity: "info" | "warning" | "error";
	readonly message: string;
	readonly instanceId?: string;
	/** Opaque component/source label — never a real filesystem path. */
	readonly source?: string;
}

/**
 * Description of a single datasource-backed source (a chunk, document, or
 * instance root). `source` is always a slash-hierarchical opaque path with no
 * `#` fragment.
 */
export interface SourceDescription {
	/** Slash-hierarchical opaque path, e.g. `/kakao/acct-1/chunks/c-42`. */
	readonly source: string;
	readonly datasourceId?: string;
	readonly skill?: string;
	readonly instanceId?: string;
	readonly chunkId?: string;
	readonly contentType?: string;
	readonly metadata: Readonly<Record<string, unknown>>;
}
