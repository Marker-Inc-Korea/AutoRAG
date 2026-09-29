/**
 * Chat-history presentation for the AI Search History popover.
 *
 * Reference: handoff README §3 "History popover" and the prototype's
 * `histGroups` (AI Finder v6.dc.html:1158-1161). The reference groups hits as
 * 오늘 / 어제 / 지난 7일 / 지난 30일; because the product keeps history
 * indefinitely, older entries get their own trailing bucket instead of being
 * mislabelled into "지난 30일".
 */

import type { ChatSummary } from "../../../shared/search-contract";

export const HISTORY_BUCKETS = ["오늘", "어제", "지난 7일", "지난 30일", "그 이전"] as const;
export type HistoryBucket = (typeof HISTORY_BUCKETS)[number];

export interface ChatHistoryEntry {
	readonly id: string;
	readonly title: string;
	readonly snippet: string;
	/** Prototype shape: "10:42" for today and yesterday, "9월 18일" before that. */
	readonly time: string;
}

export interface ChatHistoryGroup {
	readonly label: HistoryBucket;
	readonly items: readonly ChatHistoryEntry[];
}

const MS_PER_DAY = 86_400_000;
/** Yesterday and today have their own buckets, so 지난 7일 covers days 2-7 and 지난 30일 days 8-30. */
const WEEK_DAYS = 7;
const MONTH_DAYS = 30;
const NO_VALUE = "—";

function startOfDay(date: Date): number {
	return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

function daysAgo(iso: string, now: Date): number | null {
	const date = new Date(iso);
	if (Number.isNaN(date.getTime())) {
		return null;
	}
	return Math.round((startOfDay(now) - startOfDay(date)) / MS_PER_DAY);
}

export function historyBucket(updatedAt: string, now: Date): HistoryBucket {
	const days = daysAgo(updatedAt, now);
	if (days === null || days > MONTH_DAYS) return "그 이전";
	if (days > WEEK_DAYS) return "지난 30일";
	if (days > 1) return "지난 7일";
	return days === 1 ? "어제" : "오늘";
}

export function historyTimeLabel(updatedAt: string, now: Date): string {
	const date = new Date(updatedAt);
	if (Number.isNaN(date.getTime())) {
		return NO_VALUE;
	}
	const bucket = historyBucket(updatedAt, now);
	if (bucket === "오늘" || bucket === "어제") {
		return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
	}
	const day = `${date.getMonth() + 1}월 ${date.getDate()}일`;
	return date.getFullYear() === now.getFullYear() ? day : `${date.getFullYear()}년 ${day}`;
}

export function groupChatHistory(summaries: readonly ChatSummary[], now: Date): readonly ChatHistoryGroup[] {
	const grouped = new Map<HistoryBucket, ChatHistoryEntry[]>();
	for (const summary of summaries) {
		const bucket = historyBucket(summary.updatedAt, now);
		const items = grouped.get(bucket) ?? [];
		items.push({
			id: summary.id,
			title: summary.title,
			snippet: summary.snippet,
			time: historyTimeLabel(summary.updatedAt, now),
		});
		grouped.set(bucket, items);
	}
	return HISTORY_BUCKETS.filter((bucket) => grouped.has(bucket)).map((bucket) => ({
		label: bucket,
		items: grouped.get(bucket) ?? [],
	}));
}

export function historyEmptyText(query: string): string {
	const trimmed = query.trim();
	return trimmed === "" ? "저장된 대화가 없습니다." : `"${trimmed}"에 해당하는 대화가 없습니다`;
}
