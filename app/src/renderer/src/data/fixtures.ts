/**
 * Prototype fixtures — the reference's `FS` directory dictionary, verbatim
 * (docs/design-reference/ai-finder/design_handoff_ai_finder_v6/AI Finder v6.dc.html,
 * proto:728-764). Labels stay in the display domain so rows such as "14 msgs"
 * survive; the fixture source turns them into FinderEntry rows.
 *
 * This data only runs when `window.autorag.fs` is absent — vitest, a plain
 * browser, or a renderer opened before the bridge exists.
 */

import type { FixtureItem } from "./entries";

export const FIXTURE_TREE: Readonly<Record<string, readonly FixtureItem[]>> = {
	"Recents": [
		{ name: "Q3_마케팅예산_v3.xlsx", fileKind: "xlsx", dateLabel: "Sep 13, 16:48", sizeLabel: "84 KB" },
		{ name: "주간보고_0919.pptx", fileKind: "pptx", dateLabel: "Sep 19, 18:02", sizeLabel: "4.8 MB" },
		{ name: "예산승인_품의서_0915.pdf", fileKind: "pdf", dateLabel: "Sep 15, 17:10", sizeLabel: "312 KB" },
		{ name: "Q3_캠페인_브리프.docx", fileKind: "docx", dateLabel: "Sep 5, 11:20", sizeLabel: "1.1 MB" },
		{ name: "회의록_0922.md", fileKind: "md", dateLabel: "Sep 22, 10:05", sizeLabel: "6 KB" },
	],
	"Desktop": [
		{ name: "스크린샷", fileKind: "folder", dateLabel: "Sep 22, 11:40", sizeLabel: "—" },
		{ name: "주간보고_0919.pptx", fileKind: "pptx", dateLabel: "Sep 19, 18:02", sizeLabel: "4.8 MB" },
		{ name: "회의록_0922.md", fileKind: "md", dateLabel: "Sep 22, 10:05", sizeLabel: "6 KB" },
		{ name: "스크린샷 2026-09-22.png", fileKind: "png", dateLabel: "Sep 22, 11:40", sizeLabel: "1.2 MB" },
		{ name: "Q3_캠페인_브리프_초안.docx", fileKind: "docx", dateLabel: "Aug 29, 14:10", sizeLabel: "1.0 MB" },
	],
	"Desktop/스크린샷": [
		{ name: "대시보드_캡처_01.png", fileKind: "png", dateLabel: "Sep 22, 11:40", sizeLabel: "1.2 MB" },
		{ name: "대시보드_캡처_02.png", fileKind: "png", dateLabel: "Sep 22, 11:41", sizeLabel: "1.1 MB" },
	],
	"Downloads": [
		{ name: "Blue_Agency_견적서_v2.pdf", fileKind: "pdf", dateLabel: "Sep 14, 09:12", sizeLabel: "680 KB" },
		{ name: "media_plan_v3.xlsx", fileKind: "xlsx", dateLabel: "Sep 16, 15:30", sizeLabel: "142 KB" },
		{ name: "8월_광고성과_리포트.pdf", fileKind: "pdf", dateLabel: "Sep 3, 14:22", sizeLabel: "2.4 MB" },
		{ name: "brand_assets.zip", fileKind: "zip", dateLabel: "Aug 30, 10:00", sizeLabel: "48 MB" },
		{ name: "invoice_0831.pdf", fileKind: "pdf", dateLabel: "Aug 31, 18:40", sizeLabel: "96 KB" },
		{ name: "Q3_마케팅예산_v3 (1).xlsx", fileKind: "xlsx", dateLabel: "Sep 13, 16:52", sizeLabel: "84 KB" },
		{ name: "Q3_캠페인_브리프 (1).docx", fileKind: "docx", dateLabel: "Sep 5, 11:24", sizeLabel: "1.1 MB" },
	],
	"Documents": [
		{ name: "Finance", fileKind: "folder", dateLabel: "Sep 15, 17:10", sizeLabel: "—" },
		{ name: "Marketing", fileKind: "folder", dateLabel: "Sep 19, 18:02", sizeLabel: "—" },
		{ name: "Contracts", fileKind: "folder", dateLabel: "Aug 21, 13:00", sizeLabel: "—" },
		{ name: "HR", fileKind: "folder", dateLabel: "Jul 30, 09:45", sizeLabel: "—" },
		{ name: "업무인수인계.docx", fileKind: "docx", dateLabel: "Jun 2, 16:10", sizeLabel: "220 KB" },
	],
	"Documents/Finance": [
		{ name: "2026 Q2", fileKind: "folder", dateLabel: "Jun 30, 18:00", sizeLabel: "—" },
		{ name: "2026 Q3", fileKind: "folder", dateLabel: "Sep 15, 17:10", sizeLabel: "—" },
		{ name: "연간예산_2026.xlsx", fileKind: "xlsx", dateLabel: "Jan 12, 10:30", sizeLabel: "210 KB" },
		{ name: "비용집행_가이드.pdf", fileKind: "pdf", dateLabel: "Feb 3, 09:00", sizeLabel: "1.4 MB" },
	],
	"Documents/Finance/2026 Q2": [
		{ name: "Q2_결산.xlsx", fileKind: "xlsx", dateLabel: "Jul 4, 17:00", sizeLabel: "190 KB" },
	],
	"Documents/Finance/2026 Q3": [
		{ name: "벤더 견적", fileKind: "folder", dateLabel: "Sep 14, 09:12", sizeLabel: "—" },
		{ name: "Q3_마케팅예산_v1.xlsx", fileKind: "xlsx", dateLabel: "Aug 28, 11:02", sizeLabel: "78 KB" },
		{ name: "Q3_마케팅예산_v2.xlsx", fileKind: "xlsx", dateLabel: "Sep 8, 15:40", sizeLabel: "81 KB" },
		{ name: "Q3_마케팅예산_v3.xlsx", fileKind: "xlsx", dateLabel: "Sep 13, 16:48", sizeLabel: "84 KB" },
		{ name: "예산승인_품의서_0915.pdf", fileKind: "pdf", dateLabel: "Sep 15, 17:10", sizeLabel: "312 KB" },
		{ name: "집행내역_8월.csv", fileKind: "csv", dateLabel: "Sep 2, 09:30", sizeLabel: "44 KB" },
		{ name: "집행내역_9월_중간.csv", fileKind: "csv", dateLabel: "Sep 20, 12:15", sizeLabel: "39 KB" },
	],
	"Documents/Finance/2026 Q3/벤더 견적": [
		{ name: "Blue_Agency_견적서_v2.pdf", fileKind: "pdf", dateLabel: "Sep 14, 09:12", sizeLabel: "680 KB" },
	],
	"Documents/Marketing": [
		{ name: "Q3_캠페인_캘린더.xlsx", fileKind: "xlsx", dateLabel: "Sep 1, 10:00", sizeLabel: "66 KB" },
		{ name: "브랜드_가이드_2026.pdf", fileKind: "pdf", dateLabel: "Mar 3, 12:00", sizeLabel: "12 MB" },
	],
	"Documents/Contracts": [
		{ name: "Blue_Agency_MSA.pdf", fileKind: "pdf", dateLabel: "Aug 21, 13:00", sizeLabel: "840 KB" },
	],
	"Documents/HR": [
		{ name: "휴가계획_Q3.xlsx", fileKind: "xlsx", dateLabel: "Jul 30, 09:45", sizeLabel: "24 KB" },
	],
	"iCloud Drive": [
		{ name: "Personal", fileKind: "folder", dateLabel: "Sep 1, 08:00", sizeLabel: "—" },
		{ name: "메모_예산아이디어.md", fileKind: "md", dateLabel: "Aug 20, 22:14", sizeLabel: "3 KB" },
	],
	"iCloud Drive/Personal": [
	],
	"Google Drive": [
		{ name: "Marketing", fileKind: "folder", dateLabel: "Sep 5, 11:20", sizeLabel: "—" },
		{ name: "Shared with me", fileKind: "folder", dateLabel: "Sep 18, 16:00", sizeLabel: "—" },
	],
	"Google Drive/Marketing": [
		{ name: "Q3_캠페인_브리프.docx", fileKind: "docx", dateLabel: "Sep 5, 11:20", sizeLabel: "1.1 MB" },
		{ name: "KPI_트래커.xlsx", fileKind: "xlsx", dateLabel: "Sep 21, 09:00", sizeLabel: "120 KB" },
	],
	"Google Drive/Shared with me": [
		{ name: "해외법인_마케팅_가이드.pdf", fileKind: "pdf", dateLabel: "Sep 18, 16:00", sizeLabel: "2.2 MB" },
	],
	"Dropbox": [
		{ name: "agency_creatives_Q3.zip", fileKind: "zip", dateLabel: "Sep 10, 19:30", sizeLabel: "320 MB" },
	],
	"Slack": [
		{ name: "#mkt-budget", fileKind: "folder", dateLabel: "Sep 12, 14:11", sizeLabel: "—" },
		{ name: "#marketing", fileKind: "folder", dateLabel: "Sep 22, 09:40", sizeLabel: "—" },
		{ name: "#general", fileKind: "folder", dateLabel: "Sep 23, 08:30", sizeLabel: "—" },
	],
	"Slack/#mkt-budget": [
		{ name: "2026-09-12 · 예산 조정 스레드", fileKind: "slack", dateLabel: "Sep 12, 14:11", sizeLabel: "14 msgs" },
		{ name: "2026-09-08 · v2 리뷰", fileKind: "slack", dateLabel: "Sep 8, 16:02", sizeLabel: "9 msgs" },
		{ name: "2026-08-28 · v1 초안 공유", fileKind: "slack", dateLabel: "Aug 28, 11:10", sizeLabel: "5 msgs" },
	],
	"Slack/#marketing": [
		{ name: "2026-09-22 · 캠페인 킥오프", fileKind: "slack", dateLabel: "Sep 22, 09:40", sizeLabel: "22 msgs" },
	],
	"Slack/#general": [
		{ name: "2026-09-23 · 사내 공지", fileKind: "slack", dateLabel: "Sep 23, 08:30", sizeLabel: "3 msgs" },
	],
	"Gmail": [
		{ name: "Inbox", fileKind: "folder", dateLabel: "Sep 23, 08:12", sizeLabel: "—" },
		{ name: "Sent", fileKind: "folder", dateLabel: "Sep 15, 11:02", sizeLabel: "—" },
	],
	"Gmail/Inbox": [
		{ name: "RE: Q3 마케팅 예산 최종 승인 요청", fileKind: "mail", dateLabel: "Sep 15, 10:24", sizeLabel: "1 thread" },
		{ name: "[Blue Agency] 재견적서 송부", fileKind: "mail", dateLabel: "Sep 14, 09:10", sizeLabel: "1 thread" },
		{ name: "8월 광고 성과 리포트", fileKind: "mail", dateLabel: "Sep 3, 14:20", sizeLabel: "1 thread" },
	],
	"Gmail/Sent": [
		{ name: "Q3 마케팅 예산 최종 승인 요청", fileKind: "mail", dateLabel: "Sep 13, 17:02", sizeLabel: "1 thread" },
	],
	"Notion": [
		{ name: "Marketing Wiki", fileKind: "folder", dateLabel: "Aug 29, 18:20", sizeLabel: "—" },
	],
	"Notion/Marketing Wiki": [
		{ name: "Q3 Planning", fileKind: "notion", dateLabel: "Aug 29, 18:20", sizeLabel: "12 blocks" },
		{ name: "채널별 운영 원칙", fileKind: "notion", dateLabel: "Jul 2, 10:00", sizeLabel: "30 blocks" },
	],
	"Discord": [
		{ name: "#agency-sync", fileKind: "folder", dateLabel: "Sep 16, 15:28", sizeLabel: "—" },
	],
	"Discord/#agency-sync": [
		{ name: "2026-09-16 · 미디어 플랜 v3", fileKind: "discord", dateLabel: "Sep 16, 15:28", sizeLabel: "6 msgs" },
	],
	"Telegram": [
		{ name: "해외법인 공지", fileKind: "folder", dateLabel: "Sep 18, 21:05", sizeLabel: "—" },
	],
	"Telegram/해외법인 공지": [
		{ name: "2026-09-18 · Q3 budget notice", fileKind: "telegram", dateLabel: "Sep 18, 21:05", sizeLabel: "2 msgs" },
	],
};

/** Sidebar roots, in reference order. */
export const FIXTURE_ROOTS: readonly string[] = [
	"Recents",
	"Desktop",
	"Downloads",
	"Documents",
	"iCloud Drive",
	"Google Drive",
	"Dropbox",
	"Slack",
	"Gmail",
	"Notion",
	"Discord",
	"Telegram",
];

/** Pending requests in the reference's `REQ0` fixture — the sidebar bell badge. */
export const FIXTURE_PENDING_REQUESTS = 3;

/** Version-family fixture, the reference's `FAMS` shape. */
export interface FixtureFamily {
	readonly head: string;
	readonly members: readonly (readonly [
		where: string,
		name: string,
		relation: "exact" | "near" | "contains",
	])[];
}

/** The reference's `FAMS` (proto:765-772), verbatim. */
export const FIXTURE_FAMILIES: readonly FixtureFamily[] = [
	{
		head: "Documents/Finance/2026 Q3/Q3_마케팅예산_v3.xlsx",
		members: [
			["Documents/Finance/2026 Q3", "Q3_마케팅예산_v2.xlsx", "near"],
			["Documents/Finance/2026 Q3", "Q3_마케팅예산_v1.xlsx", "near"],
			["Downloads", "Q3_마케팅예산_v3 (1).xlsx", "exact"],
		],
	},
	{
		head: "Documents/Finance/2026 Q3/벤더 견적/Blue_Agency_견적서_v2.pdf",
		members: [["Downloads", "Blue_Agency_견적서_v2.pdf", "exact"]],
	},
	{
		head: "Google Drive/Marketing/Q3_캠페인_브리프.docx",
		members: [
			["Downloads", "Q3_캠페인_브리프 (1).docx", "exact"],
			["Desktop", "Q3_캠페인_브리프_초안.docx", "near"],
		],
	},
];

/** The tab the fixture session opens on — the reference's active tab. */
export const FIXTURE_INITIAL_PATH = "Documents/Finance/2026 Q3";
