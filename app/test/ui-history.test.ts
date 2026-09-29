import { describe, expect, it } from "vitest";
import type { ChatSummary } from "../src/shared/search-contract";
import { groupChatHistory, historyBucket, historyEmptyText, historyTimeLabel } from "../src/renderer/src/state/history";

/** The reference groups results as 오늘 / 어제 / 지난 7일 / 지난 30일 (handoff README §3). */
const NOW = new Date("2026-09-29T12:00:00");

function summary(id: string, updatedAt: string, title = `${id} title`, snippet = `${id} snippet`): ChatSummary {
	return { id, title, snippet, updatedAt };
}

describe("historyBucket", () => {
	it("buckets today and yesterday by calendar day, not by 24 hours", () => {
		expect(historyBucket("2026-09-29T00:01:00", NOW)).toBe("오늘");
		expect(historyBucket("2026-09-28T23:59:00", NOW)).toBe("어제");
	});

	it("buckets the rest of the past week and month", () => {
		expect(historyBucket("2026-09-27T12:00:00", NOW)).toBe("지난 7일");
		expect(historyBucket("2026-09-22T12:00:00", NOW)).toBe("지난 7일");
		expect(historyBucket("2026-09-21T12:00:00", NOW)).toBe("지난 30일");
		expect(historyBucket("2026-08-30T12:00:00", NOW)).toBe("지난 30일");
	});

	it("keeps anything older in its own bucket", () => {
		expect(historyBucket("2026-08-29T12:00:00", NOW)).toBe("그 이전");
		expect(historyBucket("2025-01-05T12:00:00", NOW)).toBe("그 이전");
		expect(historyBucket("not-a-date", NOW)).toBe("그 이전");
	});
});

describe("historyTimeLabel", () => {
	it("shows the clock time for today and yesterday", () => {
		expect(historyTimeLabel("2026-09-29T10:42:00", NOW)).toBe("10:42");
		expect(historyTimeLabel("2026-09-28T09:15:00", NOW)).toBe("09:15");
	});

	it("shows the month and day for older entries in the same year", () => {
		expect(historyTimeLabel("2026-09-21T17:30:00", NOW)).toBe("9월 21일");
		expect(historyTimeLabel("2026-07-18T17:30:00", NOW)).toBe("7월 18일");
	});

	it("adds the year only when it differs from the current one", () => {
		expect(historyTimeLabel("2025-09-18T17:30:00", NOW)).toBe("2025년 9월 18일");
	});

	it("falls back to an em dash for an unreadable timestamp", () => {
		expect(historyTimeLabel("not-a-date", NOW)).toBe("—");
	});
});

describe("groupChatHistory", () => {
	it("groups newest-first input into the reference order, dropping empty buckets", () => {
		const groups = groupChatHistory(
			[
				summary("c1", "2026-09-29T10:42:00"),
				summary("c2", "2026-09-29T09:15:00"),
				summary("c3", "2026-09-28T17:30:00"),
				summary("c4", "2026-09-25T11:02:00"),
				summary("c5", "2026-09-21T11:02:00"),
				summary("c6", "2026-08-29T11:02:00"),
			],
			NOW,
		);

		expect(groups.map((group) => group.label)).toEqual(["오늘", "어제", "지난 7일", "지난 30일", "그 이전"]);
		expect(groups.map((group) => group.items.map((item) => item.id))).toEqual([
			["c1", "c2"],
			["c3"],
			["c4"],
			["c5"],
			["c6"],
		]);
	});

	it("carries the title, snippet, and display time of each entry", () => {
		const [group] = groupChatHistory([summary("c1", "2026-09-29T10:42:00", "Q3 마케팅 예산 승인", "9월 12일 Slack에서 감액 합의")], NOW);
		expect(group?.items[0]).toEqual({
			id: "c1",
			title: "Q3 마케팅 예산 승인",
			snippet: "9월 12일 Slack에서 감액 합의",
			time: "10:42",
		});
	});

	it("returns nothing for an empty history", () => {
		expect(groupChatHistory([], NOW)).toEqual([]);
	});
});

describe("historyEmptyText", () => {
	it("names the empty store and the empty search", () => {
		expect(historyEmptyText("")).toBe("저장된 대화가 없습니다.");
		expect(historyEmptyText("   ")).toBe("저장된 대화가 없습니다.");
		expect(historyEmptyText("예산")).toBe('"예산"에 해당하는 대화가 없습니다');
	});
});
