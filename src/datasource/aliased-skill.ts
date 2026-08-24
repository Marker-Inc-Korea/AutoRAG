import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../retrieval/types.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	SourceDescription,
} from "./types.ts";

export interface AliasDatasourceSkillOptions {
	readonly alias: string;
	readonly channelIds?: readonly string[];
	readonly channelNames?: readonly string[];
}

/**
 * Gives any datasource connection an independent model-visible identity.
 * Channel allowlists are enforced after the backend search so omitted lists
 * retain the backend's default "all channels" behavior.
 */
export class AliasedDatasourceSkill implements DatasourceSkill {
	private readonly skill: DatasourceSkill;
	private readonly alias: string;
	private readonly originalId: string;
	private readonly channelIds: ReadonlySet<string>;
	private readonly channelNames: ReadonlySet<string>;

	constructor(skill: DatasourceSkill, options: AliasDatasourceSkillOptions) {
		this.skill = skill;
		this.alias = options.alias;
		this.originalId = skill.describe().name;
		this.channelIds = new Set(options.channelIds ?? []);
		this.channelNames = new Set(options.channelNames ?? []);
	}

	describe(): DatasourceSkillDescriptor {
		const descriptor = this.skill.describe();
		return {
			...descriptor,
			name: this.alias,
			id: this.alias,
			datasourceId: this.alias,
			description: `${descriptor.description} (${this.alias})`,
		};
	}

	polling() {
		return this.skill.polling();
	}

	async index(): Promise<DatasourceIndexResult> {
		const result = await this.skill.index();
		return {
			...result,
			skill: this.alias,
			diagnostics: result.diagnostics.map((diagnostic) => ({
				...diagnostic,
				source: diagnostic.source === undefined ? undefined : this.alias,
			})),
		};
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return this.skill.retrievalMethods().map((method) => ({
			describe: () => {
				const descriptor = method.describe();
				return {
					...descriptor,
					name: rewriteMethodName(descriptor.name, this.originalId, this.alias),
					datasourceId: this.alias,
				};
			},
			retrieve: async (query: string, options: RetrievalOptions) => {
				const originalOptions = {
					...options,
					scope: rewriteScope(options.scope, this.alias, this.originalId),
					allowedScopes: options.allowedScopes?.map(
						(scope) => rewriteScope(scope, this.alias, this.originalId) ?? scope,
					),
				};
				const results = await method.retrieve(query, originalOptions);
				return results.filter((result) => this.matchesChannel(result)).map((result) => this.rewriteResult(result));
			},
		}));
	}

	describeSources(): readonly SourceDescription[] {
		return this.skill.describeSources().map((source) => ({
			...source,
			source: rewriteSource(source.source, this.originalId, this.alias),
			datasourceId: this.alias,
			skill: this.alias,
			metadata: { ...source.metadata, datasourceId: this.alias },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const manifest = this.skill.skillManifest();
		const selection =
			this.channelIds.size === 0 && this.channelNames.size === 0
				? "Channel selection: all channels/chats are searchable by default."
				: `Channel selection: only these configured channels/chats are searchable: ${[
						...this.channelIds,
						...this.channelNames,
					].join(", ")}.`;
		return {
			name: `datasource-${this.alias}`,
			description: `${manifest.description} Connection alias: ${this.alias}.`,
			content: `${rewriteSource(manifest.content, this.originalId, this.alias)}\n\n${selection}`,
		};
	}

	private matchesChannel(result: RetrievalResult): boolean {
		if (this.channelIds.size === 0 && this.channelNames.size === 0) return true;
		const metadata = result.metadata;
		const id = [metadata.channelId, metadata.chatId, metadata.roomId].find(
			(value): value is string => typeof value === "string",
		);
		const name = [metadata.channelName, metadata.chatName, metadata.roomName].find(
			(value): value is string => typeof value === "string",
		);
		return (id !== undefined && this.channelIds.has(id)) || (name !== undefined && this.channelNames.has(name));
	}

	private rewriteResult(result: RetrievalResult): RetrievalResult {
		const method =
			typeof result.metadata.method === "string"
				? rewriteMethodName(result.metadata.method, this.originalId, this.alias)
				: result.metadata.method;
		return {
			...result,
			id: `${this.alias}:${result.id}`,
			source: rewriteSource(result.source, this.originalId, this.alias),
			metadata: {
				...result.metadata,
				...(method !== undefined ? { method } : {}),
				datasourceId: this.alias,
			},
		};
	}
}

function rewriteSource(value: string, originalId: string, alias: string): string {
	return value
		.replaceAll(`/${originalId}/`, `/${alias}/`)
		.replaceAll(`datasource-${originalId}`, `datasource-${alias}`);
}

function rewriteScope(scope: string | undefined, alias: string, originalId: string): string | undefined {
	return scope === undefined ? undefined : rewriteSource(scope, alias, originalId);
}

function rewriteMethodName(name: string, originalId: string, alias: string): string {
	return name.startsWith(`${originalId}-`) ? `${alias}${name.slice(originalId.length)}` : `${alias}-${name}`;
}
