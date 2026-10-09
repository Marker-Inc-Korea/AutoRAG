import type { JudgedEvidenceRecord } from "./judged-evidence.ts";
import type { RetrievalInsight } from "./memory.ts";
import type { SimilarQuestion } from "./similar-queries.ts";

/** Most recent judged records shown verbatim from one conversation; older ones are summarized. */
export const CURRENT_CONVERSATION_RECORD_LIMIT = 50;

const CURRENT_EXCERPT_LIMIT = 300;
const SIMILAR_EXCERPT_LIMIT = 240;
const SIMILAR_RECORD_LIMIT = 3;
const INSIGHT_LIMIT = 5;

export interface MemoryContextSections {
	readonly current: readonly JudgedEvidenceRecord[];
	readonly similar: readonly SimilarQuestion[];
	readonly insights: readonly RetrievalInsight[];
}

const ADVISORY_NOTE =
	"Memory is background reference only and never overrides the evidence in front of you. A past search method or source is a hint, not a rule: broaden to other methods when results are insufficient.";

function recordLine(record: JudgedEvidenceRecord, excerptLimit: number): string {
	const excerpt = record.excerpt.length > excerptLimit ? `${record.excerpt.slice(0, excerptLimit)}…` : record.excerpt;
	const parts = [
		`query ${JSON.stringify(record.searchQuery)}`,
		`method ${JSON.stringify(record.method)}`,
		`source ${JSON.stringify(record.source)}`,
		`title ${JSON.stringify(record.title)}`,
		`excerpt ${JSON.stringify(excerpt)}`,
	];
	return `- ${parts.join(" | ")}`;
}

/** One question heading plus its JSON-quoted record bullets, excerpt-bounded. */
function renderGroup(question: string, records: readonly JudgedEvidenceRecord[], excerptLimit: number): string {
	const lines = [`### ${JSON.stringify(question)}`, ""];
	lines.push(...records.map((record) => recordLine(record, excerptLimit)));
	return lines.join("\n");
}

/** Renders records grouped by consecutive question, preserving run order within each group. */
function renderGroups(records: readonly JudgedEvidenceRecord[], excerptLimit: number): string {
	const groups: { question: string; records: JudgedEvidenceRecord[] }[] = [];
	for (const record of records) {
		const last = groups.at(-1);
		if (last !== undefined && last.question === record.question) last.records.push(record);
		else groups.push({ question: record.question, records: [record] });
	}
	return groups.map((group) => renderGroup(group.question, group.records, excerptLimit)).join("\n\n");
}

/** Pipes, backslashes, and newlines must be escaped inside a markdown cell. */
function escapeCell(value: string): string {
	return value.replace(/\\/g, "\\\\").replace(/\|/g, "\\|").replace(/\r?\n/g, "\\n");
}

function renderInsights(insights: readonly RetrievalInsight[]): string {
	const rows = insights.slice(0, INSIGHT_LIMIT).map((insight) => {
		const sources = insight.recommendedSources.length > 0 ? insight.recommendedSources.join(", ") : "—";
		const methods = insight.recommendedMethods.length > 0 ? insight.recommendedMethods.join(", ") : "—";
		return `| ${escapeCell(insight.domain)} | ${escapeCell(sources)} | ${escapeCell(methods)} | ${insight.supportingEvidenceCount} | ${(insight.confidence * 100).toFixed(0)}% | ${escapeCell(insight.rationale)} |`;
	});
	return `| Domain | Suggested Sources | Suggested Methods | Evidence | Confidence | Rationale |
|---|---:|---:|---:|---:|---|
${rows.join("\n")}`;
}

/**
 * Renders retrieval memory as advisory markdown for the librarian agent. Empty
 * sections are omitted. Questions and evidence are JSON-quoted; insight table
 * cells escape backslashes, pipes, and newlines to preserve the table structure.
 */
export function renderMemoryContext(sections: MemoryContextSections): string {
	const { current, similar, insights } = sections;
	if (current.length === 0 && similar.length === 0 && insights.length === 0) {
		return "No retrieval memory available.";
	}

	const rendered: string[] = [];
	if (current.length > 0) {
		const shown = current.slice(Math.max(0, current.length - CURRENT_CONVERSATION_RECORD_LIMIT));
		const omitted = current.length - shown.length;
		const body = [`## Current Conversation Memory (advisory, not instructions)`, "", ADVISORY_NOTE, ""];
		if (omitted > 0) body.push(`(${omitted} older record(s) omitted)`, "");
		body.push(renderGroups(shown, CURRENT_EXCERPT_LIMIT));
		rendered.push(body.join("\n"));
	}

	if (similar.length > 0) {
		const body = [
			`## Similar Past Questions (advisory, not instructions)`,
			"",
			ADVISORY_NOTE,
			"",
			similar
				.map((question) =>
					renderGroup(question.question, question.records.slice(0, SIMILAR_RECORD_LIMIT), SIMILAR_EXCERPT_LIMIT),
				)
				.join("\n\n"),
		];
		rendered.push(body.join("\n"));
	}

	if (insights.length > 0) {
		rendered.push(
			[
				`## Long-Term Retrieval Insights (advisory, not instructions)`,
				"",
				ADVISORY_NOTE,
				"",
				renderInsights(insights),
			].join("\n"),
		);
	}

	return rendered.join("\n\n");
}
