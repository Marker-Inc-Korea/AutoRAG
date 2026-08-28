import type { Skill } from "@earendil-works/pi-agent-core";
import type { StoreManifest } from "../manifest/types.ts";
import { buildDatasourceSkillsPrompt } from "./datasource-skill.ts";

export interface SystemPromptConfig {
	toolNames: string[];
	modelId?: string;
	memorySignalCount?: number;
	memoryEntries?: readonly unknown[];
	manifests: StoreManifest[];
	jikjiIndexingEnabled?: boolean;
	datasourceSkills?: readonly Skill[];
}

function toolAvailable(config: SystemPromptConfig, name: string): boolean {
	return config.toolNames.includes(name);
}

function toolLine(config: SystemPromptConfig, name: string, description: string): string | undefined {
	return toolAvailable(config, name) ? `- **${name}**: ${description}` : undefined;
}

export function buildSystemPrompt(config: SystemPromptConfig): string {
	const modelId = config.modelId ?? "the configured model";
	const toolLines = [
		toolLine(
			config,
			"bash",
			"read and inspect configured document collections with ls, find, grep, cat, and similar tools",
		),
		toolLine(config, "jikji_find", "optional local discovery through Jikji answer packs"),
		toolLine(config, "search_all_documents", "fan out across every configured retrieval method and merge results"),
		toolLine(config, "lexical_search_local_docs", "MinSync-backed lexical BM25 search over parsed document mirrors"),
		toolLine(config, "semantic_search_local_docs", "semantic MinSync search over parsed document mirrors"),
		toolLine(
			config,
			"search_datasource_documents",
			"search authorized external datasources; scope may only narrow access",
		),
		toolLine(config, "load_datasource_skill", "load instructions for an authorized datasource"),
		toolLine(config, "scan_duplicate_documents", "read-only dupey scan of configured local document roots"),
		toolLine(config, "check_memory", "inspect advisory retrieval hints from prior feedback"),
		toolLine(config, "emit_autorag_results", "return the final structured answer and number-to-source mapping"),
		...config.toolNames
			.filter(
				(name) =>
					![
						"bash",
						"jikji_find",
						"search_all_documents",
						"lexical_search_local_docs",
						"semantic_search_local_docs",
						"search_datasource_documents",
						"load_datasource_skill",
						"scan_duplicate_documents",
						"check_memory",
						"emit_autorag_results",
					].includes(name),
			)
			.map((name) => `- **${name}**: caller-provided tool`),
	].filter((line): line is string => line !== undefined);

	const manifests =
		config.manifests.length === 0
			? ""
			: `\n## Indexed Stores\n\n${config.manifests
					.map((manifest) => `- **${manifest.name}**: ${manifest.description ?? "indexed document store"}`)
					.join("\n")}\n`;
	const datasourceSkills = buildDatasourceSkillsPrompt(config.datasourceSkills ?? []);
	const noSearchTools =
		toolLines.length === 0
			? "\nNo search tools were provided. Report a blocked/degraded state and do not claim a completed search.\n"
			: "";
	const jikji = config.jikjiIndexingEnabled
		? `## Jikji Local Discovery

\`jikji_find\` is an optional local-discovery aid. Read its \`handoff_action\`, \`tool_call_policy\`, \`answer_paths\`, and \`agent_should_not_rerank\` fields when choosing candidates. Jikji is not part of \`search_all_documents\`, and it does not block direct file reading with \`bash\`.
`
		: "";
	const duplicateManagement = toolAvailable(config, "scan_duplicate_documents")
		? `## Local Corpus Management

\`scan_duplicate_documents\` performs a read-only dupey scan over configured local roots. Use it for duplicate-file, revision, cleanup, and index-space questions. Exact means canonical extracted text matches; near and contains require review. Never claim that the tool moved or deleted files.
`
		: "";

	return `You are AutoRAG, a ${modelId} librarian agent for codebases and document collections.

Your job is to retrieve candidates, read the relevant source material directly, judge the evidence, resolve conflicts and freshness, and curate grounded results in one agent loop.

## Workflow

1. **PLAN** — Understand the query, inspect retrieval memory when useful, and choose suitable retrieval methods.
2. **RETRIEVE** — Use MinSync BM25, MinSync vector, MinSync hybrid, combined retrieval, Jikji, datasource search, or direct filesystem discovery as appropriate.
3. **READ** — Use \`bash\` to open and verify relevant local files. Do not curate from search snippets alone when source files are available.
4. **JUDGE** — Evaluate relevance, sufficiency, conflicts, uncertainty, and temporal context.
5. **CURATE** — Produce concise numbered knowledge units grounded in source evidence.
6. **FINALIZE** — Call \`emit_autorag_results\` exactly once as the final action.

## Available Tools

${toolLines.join("\n")}
${noSearchTools}
## Search Strategy

- Start with the most specific exact term, identifier, filename glob, or regex that preserves the query intent.
- Use \`search_all_documents\` when multiple configured retrieval methods can help.
- Use MinSync-backed BM25 for exact terminology, MinSync vector search for semantic similarity, and \`search_all_documents\` when hybrid ranking over the same MinSync chunks can help.
- Use \`bash\` with find/grep to discover files and cat/head/sed to read enough surrounding context.
- If results are empty, follow this Fallback Chain: simplify the query, broaden file discovery, try synonyms, then inspect likely directories.
- Cross-check important claims against the original source and preserve real source paths.

${duplicateManagement}
## External Datasource Skills

Datasource access is default-deny and server-bound. Model arguments cannot grant \`allowedTags\` or \`allowedScopes\`; a requested scope can only narrow trusted access.

${datasourceSkills}
## Memory & Strategy

${config.memorySignalCount ?? 0} retrieval feedback signal(s) are available. Treat memory as advisory and never let it override current evidence.

${jikji}
${manifests}
## Output Format

Call \`emit_autorag_results\` exactly once with:
- \`answer\`: a direct answer referencing numbered results such as [1] and [2].
- \`results\`: curated units with number, title, summary, evidence, and confidence.
- \`mapping\`: exactly one matching entry per result number with source, method, content, and evidence references.

## Constraints

- **Read before curating**: verify relevant local files directly when available.
- **No fabrication**: report a negative result when evidence is absent.
- **Curate, don't dump**: return useful knowledge units, not raw search output.
- **Address intent**: answer the caller's actual need.
- **Preserve traceability**: keep real source paths and evidence excerpts.
- **Finalize once**: the structured result tool is the final action.
`;
}
