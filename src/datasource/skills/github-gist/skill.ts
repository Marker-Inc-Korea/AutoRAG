/**
 * GitHub Gist datasource skill (issue #1588).
 *
 * Composition: {@link GitHubGistConnector} incrementally mirrors the
 * authenticated account's gists into the shared chunk store (lexical BM25),
 * and a {@link GistSemanticIndex} vector sidecar adds semantic retrieval
 * through the loopback embedding gateway. Embedding failures degrade to a
 * `semantic-unavailable` warning — indexing and lexical retrieval continue.
 */

import { join } from "node:path";
import type { RetrievalMethod } from "../../../retrieval/types.ts";
import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { boundDiagnosticText, sanitizeIdSegment } from "../../connector.ts";
import type { DatasourceIndexResult } from "../../types.ts";
import { GitHubGistConnector, type GitHubGistConnectorOptions } from "./connector.ts";
import {
	createGatewayGistEmbedder,
	type GatewayGistEmbedderOptions,
	GistSemanticIndex,
	GitHubGistSemanticMethod,
	type GistEmbedder,
} from "./semantic.ts";

export const GITHUB_GIST_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "github-gist",
	skillType: "github-gist",
	description: "GitHub Gist datasource (authenticated account)",
	capabilities: ["gists", "api", "polling", "incremental", "lexical", "semantic"],
	defaultTags: ["github", "gists"],
	contentType: "gist",
	manifestDescription:
		"Search the authenticated GitHub account's indexed gists — code snippets, notes, and design memos. Use for questions about content the user saved as a gist.",
	manifestNotes: [
		"Gist visibility is bounded by the configured token scopes; secret gists are included when the token allows.",
		"Semantic retrieval embeds chunks through the local loopback embedding gateway; gist content never leaves the machine for embeddings.",
	],
};

export function gistDatasourceDir(workspaceRoot: string, skillName: string, instanceId: string): string {
	return join(workspaceRoot, ".autorag", "datasources", sanitizeIdSegment(skillName), sanitizeIdSegment(instanceId));
}

export interface GitHubGistSemanticOptions {
	/** Set false to run lexical-only. Default true. */
	readonly enabled?: boolean;
	/** Injected embedder (tests); defaults to the loopback gateway embedder. */
	readonly embedder?: GistEmbedder;
	readonly embedderOptions?: GatewayGistEmbedderOptions;
	/** Vector sidecar path override; defaults under `.autorag/datasources/…`. */
	readonly statePath?: string;
}

export interface GitHubGistSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: GitHubGistConnector;
	readonly connectorOptions?: GitHubGistConnectorOptions;
	readonly semantic?: GitHubGistSemanticOptions;
}

const DEFAULT_INSTANCE_ID = "default";

export class GitHubGistSkill extends ConnectorDatasourceSkill {
	private readonly semanticIndex: GistSemanticIndex | undefined;
	private readonly semanticEmbedder: GistEmbedder | undefined;
	private readonly semanticMethod: GitHubGistSemanticMethod | undefined;

	constructor(options: GitHubGistSkillOptions = {}) {
		const { connector, connectorOptions, semantic, ...rest } = options;
		const instanceId = options.instanceId ?? DEFAULT_INSTANCE_ID;
		const statePath =
			connectorOptions?.statePath ??
			(options.workspaceRoot === undefined
				? undefined
				: join(gistDatasourceDir(options.workspaceRoot, GITHUB_GIST_SKILL_DEFINITION.skillName, instanceId), "state.json"));
		super(GITHUB_GIST_SKILL_DEFINITION, {
			...rest,
			connector:
				connector ??
				new GitHubGistConnector({
					...(connectorOptions ?? {}),
					...(statePath !== undefined && connectorOptions?.statePath === undefined ? { statePath } : {}),
				}),
		});
		if (semantic?.enabled !== false) {
			this.semanticIndex = new GistSemanticIndex({
				statePath:
					semantic?.statePath ??
					(options.workspaceRoot === undefined
						? undefined
						: join(
							gistDatasourceDir(options.workspaceRoot, GITHUB_GIST_SKILL_DEFINITION.skillName, instanceId),
							"vectors.json",
						)),
			});
			this.semanticEmbedder = semantic?.embedder ?? createGatewayGistEmbedder(semantic?.embedderOptions ?? {});
			this.semanticMethod = new GitHubGistSemanticMethod({
				skillName: this.describe().name,
				skillType: GITHUB_GIST_SKILL_DEFINITION.skillType,
				instanceId: this.instanceId,
				tags: this.describe().tags,
				store: this.store,
				index: this.semanticIndex,
				embedder: this.semanticEmbedder,
			});
		}
	}

	override async index(): Promise<DatasourceIndexResult> {
		const result = await super.index();
		if (!result.ok || this.semanticIndex === undefined || this.semanticEmbedder === undefined) return result;
		const sync = await this.semanticIndex.sync(this.store.chunks(), this.semanticEmbedder);
		if (sync.ok) return result;
		return {
			...result,
			diagnostics: [
				...result.diagnostics,
				{
					code: "datasource-index-failed" as const,
					severity: "warning" as const,
					message: boundDiagnosticText(`semantic-unavailable: ${sync.message}`),
					instanceId: this.instanceId,
					source: this.describe().name,
				},
			],
		};
	}

	override retrievalMethods(): readonly RetrievalMethod[] {
		const methods = super.retrievalMethods();
		return this.semanticMethod === undefined ? methods : [...methods, this.semanticMethod];
	}
}
