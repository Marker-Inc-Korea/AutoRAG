/**
 * Sidebar navigation — the reference's `PLACES` groups (proto:774-779).
 *
 * Icon path data and the source tiles are verbatim; `syncing` / `error` mark
 * the status dots the reference shows on Gmail, Google Drive, and Discord.
 */

import type { FileKind } from "../state/kinds";

export interface NavItem {
	readonly label: string;
	/** Stroke path for a place icon; null when the item renders a source tile. */
	readonly icon: string | null;
	readonly sourceKind: FileKind | null;
	readonly syncing: boolean;
	readonly error: boolean;
}

export interface NavGroup {
	readonly label: string;
	readonly items: readonly NavItem[];
}

function place(label: string, icon: string, status: { syncing?: boolean; error?: boolean } = {}): NavItem {
	return {
		label,
		icon,
		sourceKind: null,
		syncing: status.syncing ?? false,
		error: status.error ?? false,
	};
}

function source(label: string, sourceKind: FileKind, status: { syncing?: boolean; error?: boolean } = {}): NavItem {
	return {
		label,
		icon: null,
		sourceKind,
		syncing: status.syncing ?? false,
		error: status.error ?? false,
	};
}

export const NAV_GROUPS: readonly NavGroup[] = [
	{
		label: "Favorites",
		items: [
			place("Recents", "M12 7v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"),
			place("Desktop", "M3 5h18v11H3zM8 20h8M12 16v4"),
			place("Downloads", "M12 4v11m0 0l-4-4m4 4l4-4M5 20h14"),
			place("Documents", "M6 3h8l4 4v14H6zM14 3v4h4"),
		],
	},
	{
		label: "Cloud",
		items: [
			place("iCloud Drive", "M7 18h10a4 4 0 0 0 .5-7.97A6 6 0 0 0 6.1 11 3.5 3.5 0 0 0 7 18z"),
			place("Google Drive", "M8.5 4h7l6 10.5-3.5 6h-12l-3.5-6z", { syncing: true }),
			place("Dropbox", "M4 7l8-4 8 4-8 4zM4 7v10l8 4 8-4V7"),
		],
	},
	{
		label: "Sources",
		items: [
			source("Slack", "slack"),
			source("Gmail", "mail", { syncing: true }),
			source("Notion", "notion"),
			source("Discord", "discord", { error: true }),
			source("Telegram", "telegram"),
		],
	},
	{
		label: "People",
		items: [
			place(
				"Contacts",
				"M15 19v-1a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v1M9 10a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM21 19v-1a3.5 3.5 0 0 0-2.6-3.4M15.5 4.2a3 3 0 0 1 0 5.6",
			),
		],
	},
];

/** Footer indexing progress — the reference's fixed "Indexing · Gmail 88%". */
export const INDEXING_STATUS = { label: "Indexing · Gmail", percent: 88 } as const;
