import { Type } from "typebox";
import {
	describeTimeWindow,
	parseTimeWindow,
	type TimeWindow,
	TimeWindowError,
	windowFromOptions,
} from "../retrieval/time-window.ts";
import type { RetrievalDiagnostic, RetrievalWindowReport } from "../retrieval/types.ts";

/** The two model-facing window parameters shared by every search tool. A window only narrows what a search returns. */
export const timeWindowProperties = {
	since: Type.Optional(
		Type.String({
			description:
				"Optional ISO-8601 lower bound, inclusive: only evidence dated at or after it. A date-only value such as 2026-09-01 is that day's local midnight. Resolve relative phrases ('last month', 'this week') against the current date in the prompt.",
		}),
	),
	until: Type.Optional(
		Type.String({
			description:
				"Optional ISO-8601 upper bound, EXCLUSIVE: only evidence dated before it. 'All of September' is since 2026-09-01 and until 2026-10-01. Date-only values are local midnight.",
		}),
	),
};

/** Parse the tool's `since`/`until` once at the boundary. An invalid value throws its message verbatim; nothing is searched. */
export function parseToolWindow(params: { readonly since?: string | undefined; readonly until?: string | undefined }): {
	readonly since?: number;
	readonly until?: number;
} {
	return parseTimeWindow(params) ?? {};
}

export { TimeWindowError };

/** The notice the model reads about a windowed search: why a list is empty, and how far the search could see. */
export function windowNotice(report: RetrievalWindowReport | undefined, resultCount: number): string | undefined {
	if (report === undefined) return undefined;
	const range = describeTimeWindow(windowFromOptions(report) ?? ({} satisfies TimeWindow));
	const counts = `${report.examined} candidate(s) examined, ${report.matched} inside the range${
		report.untimed > 0 ? `, ${report.untimed} undated and excluded` : ""
	}`;
	const coverage = report.ceilingHit
		? `The over-fetch ceiling (${report.fetchDepth} candidates per method) was hit, so matches inside the range may rank below it and be unexamined: a thin or empty result is not proof that nothing exists in the range. Narrow the query or the range.`
		: "Every candidate the methods returned was examined.";
	if (resultCount === 0) return `Nothing in this time range ${range}: ${counts}. ${coverage}`;
	return `Time window ${range}: ${counts}. ${coverage}`;
}

/**
 * Diagnostics line for a search tool. Window diagnostics carry their reason verbatim (which method could not date its
 * results and why); the rest stay `source:code`.
 */
export function diagnosticSummary(diagnostics: readonly RetrievalDiagnostic[], fallbackSource: string): string {
	if (diagnostics.length === 0) return "";
	const summary = `Diagnostics: ${diagnostics.map((d) => `${d.source ?? fallbackSource}:${d.code}`).join(", ")}`;
	const reasons = diagnostics.filter((d) => d.code === "time-window-untimed").map((d) => d.reason ?? d.message);
	return `\n\n${[summary, ...reasons].join("\n")}`;
}
