/**
 * LLM-facing formatting for web search responses.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT) `web/search/index.ts`
 * (formatForLLM) and its prompts/tools + prompts/system web-search markdown.
 */
import type { SearchResponse } from "./types.ts";

/**
 * System prompt passed to LLM-mediated providers. Kept for forward
 * compatibility — every provider AutoRAG ships today returns sources
 * directly, but the SearchParams contract carries it.
 */
export const WEB_SEARCH_SYSTEM_PROMPT = `Web research assistant: accurate, well-sourced, comprehensive answers.

<priorities>
1. Accuracy > speed; verify claims across multiple sources when possible.
2. Primary > secondary: official docs, papers, announcements > blog summaries.
3. Recency matters: note publication dates; prefer recent sources for time-sensitive topics.
4. Uncertainty: distinguish confirmed facts from inferences.
</priorities>

<synthesis>
- Direct answer first; then supporting evidence.
- Quote or paraphrase specific sources; no vague attributions.
- Source conflicts: acknowledge discrepancy; identify the more authoritative source.
- Technical topics: prefer official documentation and specifications.
- News/events: prefer primary reporting over aggregators.
- Concrete data: version numbers, dates, exact figures, code snippets, specific examples.
</synthesis>

<format>
- Thorough, in-depth coverage with specific evidence; no surface-level summaries.
- Omit filler and unnecessary hedging; do NOT sacrifice detail for brevity.
- Include publication dates when recency affects relevance.
- Clear sections for multiple aspects.
- Cite sources inline using provided search results.
</format>`;

/**
 * Tool description shared by the `web_search` agent tool and CLI surfaces.
 * Ported from oh-my-pi prompts/tools/web-search.md.
 */
export const WEB_SEARCH_TOOL_DESCRIPTION = `Web search: current information beyond knowledge cutoff.

- SHOULD prefer primary sources (papers, official docs); corroborate key claims with multiple sources.
- MUST link cited sources in final response.
- NEVER use for programmatically accessible content or known URLs (GitHub repos/issues, known arXiv papers, Wikipedia pages, official docs) — use web_fetch on the URL directly.
- \`query\`: every provider supports Google-style \`site:\`/\`-site:\`, \`after:\`/\`before:\` (YYYY-MM-DD), \`inurl:\`, \`intitle:\`, \`filetype:\`, \`"exact phrase"\`, \`-term\`, \`OR\`. Map constraints to native filters when available; otherwise results are filtered leniently. If a constraint matches nothing, it is relaxed and reported; zero results are never forced.`;

/** Format an age in seconds as a compact relative string ("45s", "2h", "3d"). */
export function formatAge(ageSeconds: number | undefined): string {
	if (ageSeconds === undefined || !Number.isFinite(ageSeconds) || ageSeconds < 0) return "";
	if (ageSeconds < 90) return `${Math.round(ageSeconds)}s ago`;
	const minutes = ageSeconds / 60;
	if (minutes < 90) return `${Math.round(minutes)}m ago`;
	const hours = minutes / 60;
	if (hours < 48) return `${Math.round(hours)}h ago`;
	const days = hours / 24;
	if (days < 60) return `${Math.round(days)}d ago`;
	const months = days / 30;
	if (months < 24) return `${Math.round(months)}mo ago`;
	return `${Math.round(days / 365)}y ago`;
}

/** Truncate text for tool output */
function truncateText(text: string, maxLen: number): string {
	if (text.length <= maxLen) return text;
	return `${text.slice(0, Math.max(0, maxLen - 1))}…`;
}

function formatCount(label: string, count: number): string {
	return `${count} ${label}${count === 1 ? "" : "s"}`;
}

/** Format response for LLM consumption. `notes` lead the output (e.g. relaxed-constraint warnings). */
export function formatForLLM(response: SearchResponse, notes: readonly string[] = []): string {
	const parts: string[] = [];
	for (const note of notes) {
		parts.push(`Note: ${note}`);
	}

	if (response.answer) {
		parts.push(response.answer);
		if (response.sources.length > 0) {
			parts.push("\n## Sources");
			parts.push(formatCount("source", response.sources.length));
		}
	}

	for (const [i, src] of response.sources.entries()) {
		const age = formatAge(src.ageSeconds) || src.publishedDate;
		const agePart = age ? ` (${age})` : "";
		parts.push(`[${i + 1}] ${src.title}${agePart}\n    ${src.url}`);
		if (src.snippet) {
			parts.push(`    ${truncateText(src.snippet, 240)}`);
		}
	}

	if (response.citations && response.citations.length > 0) {
		parts.push("\n## Citations");
		parts.push(formatCount("citation", response.citations.length));
		for (const [i, citation] of response.citations.entries()) {
			const title = citation.title || citation.url;
			parts.push(`[${i + 1}] ${title}\n    ${citation.url}`);
			if (citation.citedText) {
				parts.push(`    ${truncateText(citation.citedText, 240)}`);
			}
		}
	}

	if (response.relatedQuestions && response.relatedQuestions.length > 0) {
		parts.push("\n## Related");
		parts.push(formatCount("question", response.relatedQuestions.length));
		for (const q of response.relatedQuestions) {
			parts.push(`- ${q}`);
		}
	}

	if (response.searchQueries && response.searchQueries.length > 0) {
		parts.push(`Search queries: ${response.searchQueries.length}`);
		for (const query of response.searchQueries.slice(0, 3)) {
			parts.push(`- ${truncateText(query, 120)}`);
		}
	}

	return parts.join("\n");
}

export function hasRenderableSearchContent(response: SearchResponse): boolean {
	if (response.answer?.trim()) return true;
	if (response.sources.length > 0) return true;
	if (response.citations?.length) return true;
	if (response.relatedQuestions?.some((question) => question.trim())) return true;
	if (response.searchQueries?.some((query) => query.trim())) return true;
	return false;
}
