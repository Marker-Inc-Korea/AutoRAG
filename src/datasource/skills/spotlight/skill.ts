import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { boundDiagnosticText, connectorFailureToDiagnosticCode } from "../../connector.ts";
import { datasourceCliError } from "../../errors.ts";
import { datasourceSourcePath, matchesDatasourceScope, normalizeVirtualPath } from "../../scope.ts";
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
import { SpotlightConnector, type SpotlightConnectorOptions } from "./connector.ts";

const DEFAULT_TAGS = ["spotlight", "files"] as const;
const DEFAULT_TOP_K = 20;
const LIVE_NOTE = "live-search datasource: macOS owns the Spotlight index; AutoRAG stores no snapshot";

export interface SpotlightSkillOptions {
	/** Config alias used as the independent datasource/skill identity. */
	readonly skillName?: string;
	readonly instanceId?: string;
	readonly instances?: readonly string[];
	readonly tags?: readonly string[];
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: SpotlightConnector;
	readonly connectorOptions?: SpotlightConnectorOptions;
}

/** Live macOS Spotlight datasource. Refresh probes readiness; retrieval runs `mdfind`. */
export class SpotlightSkill implements DatasourceSkill {
	private readonly connector: SpotlightConnector;
	private readonly skillName: string;
	private readonly instanceId: string;
	private readonly instances: readonly string[];
	private readonly tags: readonly string[];
	private lastIndexedAt: number | undefined;
	private lastError: string | undefined;

	constructor(options: SpotlightSkillOptions = {}) {
		this.skillName = options.skillName ?? "spotlight";
		this.instanceId = options.instanceId ?? "default";
		this.instances =
			options.instances !== undefined && options.instances.length > 0 ? options.instances : [this.instanceId];
		this.tags = options.tags ?? DEFAULT_TAGS;
		this.connector = options.connector ?? new SpotlightConnector(options.connectorOptions ?? {});
	}

	describe(): DatasourceSkillDescriptor {
		return {
			name: this.skillName,
			id: this.skillName,
			type: "spotlight-search",
			description: "Live macOS Spotlight search via the built-in mdfind CLI",
			capabilities: ["files", "macos-only", "local-search", "live-search", "scoped", "external-cli"],
			tags: this.tags,
			status: "active",
			requiresExternalCli: true,
			datasourceId: this.skillName,
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
			const result = await this.connector.probe();
			if (!result.ok) {
				const code = connectorFailureToDiagnosticCode(result.reason);
				const message = boundDiagnosticText(result.message ?? result.reason);
				this.lastError = message;
				return this.fail(code, message);
			}
			this.lastError = undefined;
			this.lastIndexedAt = Date.now();
			return {
				ok: true,
				instanceId: this.instanceId,
				skill: this.skillName,
				chunkCount: 0,
				indexedAt: this.lastIndexedAt,
				diagnostics: [this.note(LIVE_NOTE, "datasource-empty", "info")],
			};
		} catch (error) {
			const message = boundDiagnosticText(error instanceof Error ? error.message : "spotlight probe failed");
			this.lastError = message;
			return this.fail("datasource-unavailable", message);
		}
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return [new SpotlightLiveMethod(this)];
	}

	describeSources(): readonly SourceDescription[] {
		return this.instances.map((instanceId) => ({
			source: datasourceSourcePath(this.skillName, instanceId),
			datasourceId: this.skillName,
			skill: this.skillName,
			instanceId,
			contentType: "document",
			metadata: { datasourceId: this.skillName, instanceId, tags: this.tags },
		}));
	}

	skillManifest(): DatasourceSkillManifest {
		const scopes = this.instances
			.map((instanceId) => `- \`${datasourceSourcePath(this.skillName, instanceId)}\``)
			.join("\n");
		return {
			name: `datasource-${this.skillName}`,
			description: "Search the current macOS Spotlight index directly. AutoRAG keeps no file snapshot.",
			content: [
				"# macOS Spotlight (live mdfind search)",
				"",
				"Search files indexed by macOS Spotlight via the built-in `mdfind` CLI. macOS owns and refreshes the index; AutoRAG does not copy files into a local snapshot.",
				"",
				"## Platform requirements",
				"- macOS only: this datasource is unavailable on any other platform.",
				"- Spotlight indexing must be enabled (`sudo mdutil -i on /`; check with `mdutil -s /`).",
				"- Full Disk Access may be required to read protected hits. Grant it under System Settings -> Privacy & Security -> Full Disk Access, then restart the host app.",
				"",
				"## How to search",
				`Call the dedicated \`${datasourceSearchToolName(this.skillName)}\` tool with a query and optional narrowing scope. Available scopes:`,
				scopes,
				"Each configured Spotlight filter is AND-ed with the user query. Only the requested hits are read; refresh performs a readiness probe without running `mdfind`.",
				"Result metadata carries the real absolute file path (`path`) so hits are traceable back to disk. If paths must not leave this machine, run AutoRAG with a local LLM.",
				"",
				"## Native CLI",
				"- `mdfind <query>` searches Spotlight; `mdfind -onlyin <dir> <query>` scopes to a directory; `mdfind -name <text>` matches file names.",
				"- `mdls <file>` shows Spotlight metadata for one file.",
				"- `mdutil -s /` checks index status; `sudo mdutil -i on /` enables indexing.",
			].join("\n"),
		};
	}

	async search(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		const scope = options.scope?.trim() ? normalizeVirtualPath(options.scope) : undefined;
		if (scope !== undefined) {
			const chunksRoot = normalizeVirtualPath(`/${this.skillName}/${this.instanceId}/chunks`);
			const scopeProbe = datasourceSourcePath(this.skillName, this.instanceId, "__scope_probe__");
			if (
				!matchesDatasourceScope(chunksRoot, scope) &&
				!matchesDatasourceScope(scopeProbe, scope) &&
				!scope.startsWith(`${chunksRoot}/`)
			) {
				return [];
			}
		}
		const topK = options.topK ?? DEFAULT_TOP_K;
		const result = await this.connector.search(trimmed, topK, (docId) =>
			matchesDatasourceScope(datasourceSourcePath(this.skillName, this.instanceId, docId), scope),
		);
		if (!result.ok) {
			throw datasourceCliError(this.skillName, "search", {
				reason: result.reason,
				stderr: result.stderr ?? result.message,
				stdout: result.stdout,
				code: result.code,
			});
		}
		const hits: RetrievalResult[] = [];
		for (const [index, document] of result.documents.entries()) {
			const source = datasourceSourcePath(this.skillName, this.instanceId, document.docId);
			if (!matchesDatasourceScope(source, options.scope)) continue;
			hits.push({
				id: `${this.skillName}:${this.instanceId}:${document.docId}`,
				content: document.content,
				source,
				score: 1 / (index + 1),
				metadata: {
					...document.metadata,
					method: `${this.skillName}-live`,
					datasourceId: this.skillName,
					instanceId: this.instanceId,
					chunkId: document.docId,
					...(document.title !== undefined ? { title: document.title } : {}),
					...(document.hierarchy !== undefined && document.hierarchy.length > 0
						? { hierarchy: document.hierarchy.join("/") }
						: {}),
					...(document.publishedAt !== undefined ? { publishedAt: document.publishedAt } : {}),
					...(result.warnings !== undefined && result.warnings.length > 0 ? { warnings: result.warnings } : {}),
				},
			});
			if (hits.length >= topK) break;
		}
		return hits;
	}

	private fail(code: DatasourceDiagnosticCode, message: string): DatasourceIndexResult {
		return {
			ok: false,
			instanceId: this.instanceId,
			skill: this.skillName,
			indexedAt: Date.now(),
			diagnostics: [
				this.note(
					message,
					code,
					code === "datasource-unavailable" || code === "datasource-empty" ? "warning" : "error",
				),
			],
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
		return { code, severity, message, instanceId: this.instanceId, source: this.skillName };
	}
}

class SpotlightLiveMethod implements RetrievalMethod {
	private readonly skill: SpotlightSkill;

	constructor(skill: SpotlightSkill) {
		this.skill = skill;
	}

	describe(): RetrievalMethodDescriptor {
		const descriptor = this.skill.describe();
		return {
			name: `${descriptor.datasourceId ?? "spotlight"}-live`,
			type: "remote",
			description: "Live macOS Spotlight search via mdfind",
			status: "active",
			capabilities: ["live-search", "scoped", "external-cli", "path-opaque-sources"],
			datasourceId: descriptor.datasourceId,
			tags: descriptor.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return this.skill.search(query, options);
	}
}
