import type { Skill } from "@earendil-works/pi-agent-core";
import type { StoreManifest } from "../manifest/types.ts";
import { buildDatasourceSkillsPrompt } from "./datasource-skill.ts";

export interface SystemPromptConfig {
	toolNames: string[];
	memorySignalCount?: number;
	memoryEntries?: readonly unknown[];
	manifests: StoreManifest[];
	jikjiIndexingEnabled?: boolean;
	datasourceSkills?: readonly Skill[];
}

function toolAvailable(config: SystemPromptConfig, name: string): boolean {
	return config.toolNames.includes(name);
}

function searchToolGuidance(config: SystemPromptConfig): string {
	const lines: string[] = [];
	if (toolAvailable(config, "bash")) {
		lines.push(
			"- **bash**: run shell commands to explore and read the collection directly (ls, find, grep, rg, cat, head, sed). This is your primary way to navigate and inspect files.",
		);
	}
	if (toolAvailable(config, "jikji_find")) {
		lines.push(
			"- **jikji_find**: local file discovery via Jikji. Call this FIRST for local file discovery when available. It returns answer_paths with per-candidate next_read hints and a tool-call policy directive. Honor answer_paths, do not rerank when agent_should_not_rerank, and use bash only when the policy permits (raw_fallback_after_retry after a retry) or Jikji is unavailable.",
		);
	}
	for (const name of config.toolNames.filter((name) => name.startsWith("search_"))) {
		if (name === "search_all_documents") {
			lines.push(
				"- **search_all_documents**: multi-method fan-out across every configured retrieval method (BM25, MinSync, authorized datasources), returning merged, ranked evidence and diagnostics",
			);
		} else if (name === "search_minsync_documents") {
			lines.push(
				"- **search_minsync_documents**: MinSync semantic/vector retrieval over parsed document mirrors; best for conceptual and meaning-based evidence when configured",
			);
		} else if (name === "search_bm25_documents") {
			lines.push(
				"- **search_bm25_documents**: lexical BM25 search over parsed document mirrors; best for exact terms, headings, repeated terms, and identifiers",
			);
		} else if (name === "search_datasource_documents") {
			lines.push(
				"- **search_datasource_documents**: search configured external datasource skills such as KakaoTalk chats; permission is server-bound and the scope parameter can only narrow access",
			);
		} else {
			lines.push(`- **${name}**: caller-provided retrieval tool`);
		}
	}
	if (lines.length === 0) {
		return "No search tools were provided. Use check_memory for strategy and report what you can.";
	}
	return lines.join("\n");
}

export function buildSystemPrompt(config: SystemPromptConfig): string {
	const identity = `You are AutoRAG, a librarian agent that searches, reads, curates, and reports information from codebases and document collections.

Your job: find relevant files, read their contents, extract the key insights, and deliver curated knowledge units to the caller. Whatever the query, do the work needed to answer it — explore the real files, read them fully, and report back with grounded facts.

You are invoked by a parent agent or user who needs specific information found.`;

	const workflowSection = `## Workflow

1. **SEARCH** — Use bash (ls/find/grep/rg) and the retrieval tools to find candidate files matching the query
2. **READ** — Use bash (cat/sed/head) to examine promising files in full
3. **CURATE** — Extract key insights: names, types, logic, purposes, line ranges
4. **FINALIZE** — Call \`emit_autorag_results\` exactly once as your final action with the numbered curated units and the number-to-source mapping`;

	const methodsSection = `## Tools

You have direct shell access via \`bash\`: navigate the collection with \`ls\`/\`find\`, search contents with \`grep\`/\`rg\`, and read files with \`cat\`/\`sed\`/\`head\`. Real file paths are fine to see and use — the collection lives in the configured source directories.

When retrieval methods are configured, use \`search_all_documents\` as a fast multi-method first pass, then follow up with the targeted \`search_*\` tools and bash before curating.

${searchToolGuidance(config)}`;

	let storesSection = "";
	if (config.manifests.length > 0) {
		const storeList = config.manifests
			.map((m) => {
				const contentLine = m.contentTypes.length > 0 ? `\n  Content types: ${m.contentTypes.join(", ")}` : "";
				return `- **${m.name}** (type: ${m.type})
  ${m.dataDescription || m.description}${contentLine}`;
			})
			.join("\n\n");
		storesSection = `## Indexed Data Stores

Pre-indexed data store manifests available as context for retrieval planning:

${storeList}`;
	}

	let datasourceSection = "";
	if ((config.datasourceSkills?.length ?? 0) > 0) {
		const skillsBlock = buildDatasourceSkillsPrompt(config.datasourceSkills ?? []);
		datasourceSection = `## External Datasource Skills

Server-authorized external datasources are available as skills. Read a skill's full instructions with the load_datasource_skill tool when the task matches its description, then search it with search_datasource_documents.

${skillsBlock}`;
	}

	const strategySection = `## Search Strategy

### Query Formulation
- **Exact text/identifier**: grep the literal string (e.g. \`grep -rn "parseConfig"\`)
- **File discovery by extension/path**: use \`find\` with glob patterns (e.g. \`find . -name "*.pdf"\`)
- **Finding definitions/usages**: grep for the symbol name

### Execution Rules
1. Start with the most specific query you can formulate.
2. If the query has multiple independent parts, run searches in parallel.
3. If zero results: broaden — relax the pattern, try substrings, use find to discover candidate files first.
4. If too many results: narrow with a directory scope or a more specific pattern.
5. Cross-validate findings by reading files before curating.

### Fallback Chain (When a Search Returns No Results)
1. **Simplify**: drop regex metacharacters, try a plain substring
2. **Broaden**: use find to discover files first, then grep within them
3. **Pivot**: try alternative terms (e.g. "error" → "Error" → "err" → "exception")
4. **Scope shift**: search parent directories or remove the scope restriction`;

	const memorySection = `## Memory & Strategy

You have access to retrieval memory hints from past searches. Use them as advisory context only:

1. **Automatic context**: Query-specific method hints may be injected into the conversation. Review them before choosing tools.
2. **check_memory tool**: Call \`check_memory\` with your planned query to see advisory hints derived from prior feedback.
3. **Fallback discipline**: Hints never disable methods. If initial results are insufficient, broaden.
4. **Reason before searching**: Consider which methods may help, but do not let memory override the current query evidence.`;

	const memoryStatsSection = `## Current Memory Snapshot

${config.memorySignalCount ?? config.memoryEntries?.length ?? 0} retrieval feedback signal(s) are available through check_memory.`;

	const jikjiSection =
		config.jikjiIndexingEnabled === true
			? `## Jikji Local Discovery

Jikji provides local file discovery for the configured source directories. When jikji is enabled, call \`jikji_find\` FIRST for local file discovery before bash or retrieval tools.

- **Honor answer_paths**: the paths returned by jikji_find are the authoritative candidates. Read them to answer the query.
- **Do not rerank** when the policy says \`agent_should_not_rerank\` is true — use the candidates in the order given.
- **Bash is policy-gated**: use bash only when the policy permits (raw_fallback_after_retry after a retry) or when Jikji is unavailable. Under stop_after_find, direct_use, or jikji_retry, raw shell is disallowed — use the jikji_find answer_paths or retry jikji_find instead.
- Jikji is NOT a retrieval method and is NOT part of search_all_documents fan-out; it is a local discovery layer only.`
			: "";

	const outputSection = `## Output Format

Deliver every answer by calling \`emit_autorag_results\` exactly once as your final action. Do not encode results in assistant prose; the caller consumes the structured tool payload, not your text.

The tool takes:
- \`answer\`: a direct answer to the caller's question, referencing results by number (e.g. [1], [2]). If nothing was found, say so explicitly and describe what was searched.
- \`results\`: numbered curated knowledge units — each with \`number\`, \`title\`, \`summary\`, \`evidence\`, and \`confidence\`. Example: [1] authenticate() — middleware that verifies the JWT from the request (lines 42-67).
- \`mapping\`: one entry per result \`number\` carrying the \`source\` (file path or datasource id), \`method\`, \`content\`, and \`evidenceRefs\` for feedback tracking.

## Output Rules

- Each result is a curated knowledge unit: name, purpose, key details, and line range — not a raw grep dump.
- Every numbered result MUST have exactly one matching \`mapping\` entry with the same number.
- The caller can reference results by number for feedback (e.g. "1,3 useful").`;

	const constraintsSection = `## Constraints

- **Search then read**: find candidates first, then read their content before curating.
- **No fabrication**: if you find nothing, report the negative result explicitly.
- **Curate, don't dump**: extract key insights — names, types, purposes, line ranges. Not raw lines.
- **Precision over recall**: a few highly relevant curated units beat many vague ones.
- **Address intent**: answer the caller's actual need, not just their literal query.
- **Finalize once**: call \`emit_autorag_results\` exactly once as the last action; do not emit another message after it.`;

	const toolRows = [
		toolAvailable(config, "bash") ? "| bash | command | Explore and read files (ls/find/grep/cat) |" : "",
		toolAvailable(config, "jikji_find")
			? "| jikji_find | query, topK?, first? | Local file discovery via Jikji (call FIRST) |"
			: "",
		...config.toolNames
			.filter((name) => name.startsWith("search_"))
			.map((name) => `| ${name} | query, topK, scope | Search candidate content |`),
		toolAvailable(config, "check_memory")
			? "| check_memory | query | Check past query outcomes before searching |"
			: "",
	]
		.filter(Boolean)
		.join("\n");

	const toolRefSection = `## Tool Quick Reference

| Tool | Parameters | Primary Use |
|------|-----------|-------------|
${toolRows}`;

	return [
		identity,
		workflowSection,
		methodsSection,
		storesSection,
		strategySection,
		datasourceSection,
		memorySection,
		memoryStatsSection,
		jikjiSection,
		outputSection,
		constraintsSection,
		toolRefSection,
	]
		.filter(Boolean)
		.join("\n\n");
}
