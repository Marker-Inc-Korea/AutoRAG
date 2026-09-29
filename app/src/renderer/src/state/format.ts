/**
 * Display formatting for list metadata, status text, and toasts.
 *
 * Every literal here is the reference's own string (handoff README §2). The
 * date and size shapes match the prototype fixtures: "Sep 13, 16:48", "84 KB",
 * "4.8 MB" — KB is whole, MB and above carry one decimal below 10.
 */

import { basename } from "./paths";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;
const UNITS = ["KB", "MB", "GB", "TB"] as const;
/** Folders and unknown values render as an em dash, like the reference. */
const NO_VALUE = "—";

function pad(value: number): string {
	return String(value).padStart(2, "0");
}

export function formatModified(iso: string): string {
	const date = new Date(iso);
	if (Number.isNaN(date.getTime())) {
		return NO_VALUE;
	}
	return `${MONTHS[date.getMonth()] ?? NO_VALUE} ${date.getDate()}, ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** Sortable stamp for a "Sep 13, 16:48" label; 0 when it cannot be read. */
export function parseModifiedLabel(label: string): number {
	const match = /^(\w{3}) (\d+), (\d+):(\d+)$/.exec(label);
	if (match === null) {
		return 0;
	}
	const month = MONTHS.indexOf((match[1] ?? "") as (typeof MONTHS)[number]) + 1;
	const day = Number(match[2]);
	const hour = Number(match[3]);
	const minute = Number(match[4]);
	return ((month * 100 + day) * 100 + hour) * 100 + minute;
}

export function formatSize(bytes: number | null): string {
	if (bytes === null) {
		return NO_VALUE;
	}
	if (bytes < 1024) {
		return `${bytes} B`;
	}
	let value = bytes / 1024;
	let unit = 0;
	while (value >= 1024 && unit < UNITS.length - 1) {
		value /= 1024;
		unit += 1;
	}
	const label = unit >= 1 && value < 10 ? value.toFixed(1) : String(Math.round(value));
	return `${label} ${UNITS[unit] ?? "KB"}`;
}

/** Sortable byte count for a "84 KB" label. */
export function parseSizeLabel(label: string): number {
	const match = /^([\d.]+)\s*(B|KB|MB|GB|TB)$/.exec(label.trim());
	if (match === null) {
		return 0;
	}
	const scale: Record<string, number> = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 };
	return Number(match[1]) * (scale[match[2] ?? "B"] ?? 1);
}

export function statusBarText(itemCount: number, selectedCount: number): string {
	return `${itemCount} items${selectedCount > 0 ? ` · ${selectedCount} selected` : ""}`;
}

export function searchSummaryText(count: number): string {
	return `${count} results across all locations`;
}

export function trashToast(paths: readonly string[]): string {
	if (paths.length === 1) {
		return `${basename(paths[0] ?? "")} — 휴지통으로 이동했습니다`;
	}
	return `${paths.length}개 항목을 휴지통으로 이동했습니다`;
}

export function indexToast(name: string, included: boolean): string {
	return included ? `${name} — 인덱싱에 포함했습니다` : `${name} — 인덱싱에서 제외했습니다`;
}

export function emptyStateText(query: string): string {
	const trimmed = query.trim();
	return trimmed === "" ? "빈 폴더" : `"${trimmed}"와 일치하는 파일이 없습니다`;
}
