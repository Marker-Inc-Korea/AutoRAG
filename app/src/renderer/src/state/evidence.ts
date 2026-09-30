/**
 * Evidence-panel view model (DESIGN.md §5 Evidence panel; handoff README §1).
 *
 * Pure helpers for the bottom evidence strip: breadcrumb segments of the
 * source path, tile kind inference from the evidence title, thumbs-up/down
 * toggle state, and the reference toast copy.
 */

import { type FileKind, kindFromExtension } from "./kinds";
import { pathSegments } from "./paths";

export interface EvidenceCrumb {
	readonly label: string;
	/** The file/id tail renders in 600 weight; folders render 400. */
	readonly last: boolean;
}

/** Breadcrumb segments of a source path or a datasource virtual id. */
export function evidenceCrumbs(source: string): readonly EvidenceCrumb[] {
	const segments = pathSegments(source);
	return segments.map((label, index) => ({ label, last: index === segments.length - 1 }));
}

/** Confidence as a percentage, shown right of the breadcrumb row. */
export function evidenceDetail(confidence: number): string {
	return `관련도 ${Math.round(confidence * 100)}%`;
}

const EXTENSION = /\.([A-Za-z0-9]{1,6})$/u;

/**
 * Tile kind for an evidence entry: the title is the file name for local
 * results; datasource titles (subjects, page names) fall back to the
 * generic document tile.
 */
export function evidenceKind(title: string): FileKind {
	const match = EXTENSION.exec(title.trim());
	return match === null ? "md" : kindFromExtension(match[1]?.toLowerCase() ?? "");
}

export type EvidenceFeedback = "up" | "down" | null;

/** Toggle per the reference: same vote clears it; the opposite side replaces it. */
export function nextEvidenceFeedback(current: EvidenceFeedback, pick: "up" | "down"): EvidenceFeedback {
	return current === pick ? null : pick;
}

/** Reference toast copy (`proto:1197`, strip when the vote clears). */
export function feedbackToastText(number: number, value: "up" | "down"): string {
	return value === "up"
		? `근거 ${number} — 도움이 됨으로 기록했습니다`
		: `근거 ${number} — 다음 검색부터 우선순위를 낮춥니다`;
}
