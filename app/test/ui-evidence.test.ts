import { describe, expect, it } from "vitest";
import {
	evidenceCrumbs,
	evidenceDetail,
	evidenceKind,
	feedbackToastText,
	nextEvidenceFeedback,
} from "../src/renderer/src/state/evidence";

describe("evidenceCrumbs", () => {
	it("splits an OS-absolute path and marks the last segment bold", () => {
		expect(evidenceCrumbs("/Users/me/Documents/Finance/Q3_마케팅예산_v3.xlsx")).toEqual([
			{ label: "Users", last: false },
			{ label: "me", last: false },
			{ label: "Documents", last: false },
			{ label: "Finance", last: false },
			{ label: "Q3_마케팅예산_v3.xlsx", last: true },
		]);
	});

	it("handles datasource virtual ids the same way (display only)", () => {
		expect(evidenceCrumbs("/kakao/qa-instance/chunks/42")).toEqual([
			{ label: "kakao", last: false },
			{ label: "qa-instance", last: false },
			{ label: "chunks", last: false },
			{ label: "42", last: true },
		]);
	});
});

describe("evidenceDetail", () => {
	it("shows the confidence as a percentage", () => {
		expect(evidenceDetail(0.8735)).toBe("관련도 87%");
		expect(evidenceDetail(1)).toBe("관련도 100%");
	});
});

describe("evidenceKind", () => {
	it("maps a file-like title to its kind tile", () => {
		expect(evidenceKind("Q3_마케팅예산_v3.xlsx")).toBe("xlsx");
		expect(evidenceKind("예산승인_품의서_0915.pdf")).toBe("pdf");
	});

	it("falls back to the document tile for names without a known extension", () => {
		expect(evidenceKind("RE: Q3 마케팅 예산 최종 승인 요청")).toBe("md");
	});
});

describe("nextEvidenceFeedback", () => {
	it("toggles the same value off and switches between up and down", () => {
		expect(nextEvidenceFeedback(null, "up")).toBe("up");
		expect(nextEvidenceFeedback("up", "up")).toBeNull();
		expect(nextEvidenceFeedback("up", "down")).toBe("down");
		expect(nextEvidenceFeedback("down", "down")).toBeNull();
		expect(nextEvidenceFeedback(null, "down")).toBe("down");
	});
});

describe("feedbackToastText", () => {
	it("matches the reference toasts by number and value", () => {
		expect(feedbackToastText(3, "up")).toBe("근거 3 — 도움이 됨으로 기록했습니다");
		expect(feedbackToastText(3, "down")).toBe("근거 3 — 다음 검색부터 우선순위를 낮춥니다");
	});
});
