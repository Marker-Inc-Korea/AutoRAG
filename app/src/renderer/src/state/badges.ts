/**
 * Badge state → documented visual state (DESIGN.md §5 PillBadge).
 *
 * Pure mapping: labels, tone class, dot token, and the tooltip that doubles as
 * the accessible name. Every label ships with its dot, and inherited access
 * carries the " · 상위 폴더 설정" suffix (DESIGN.md §8 constraint 3).
 */

export interface BadgeVisual {
	readonly label: string;
	readonly toneClass: string;
	readonly dotToken: string | null;
	readonly hollowDot: boolean;
	readonly spinning: boolean;
	readonly title: string;
}

export type IndexState = "included" | "excluded" | "failed" | "retrying";

export function indexBadge(state: IndexState, reason = ""): BadgeVisual {
	switch (state) {
		case "included":
			return {
				label: "포함",
				toneClass: "pill--included",
				dotToken: "--index-dot-included",
				hollowDot: false,
				spinning: false,
				title: "에이전트 검색에 포함됨 · 클릭하면 제외",
			};
		case "excluded":
			return {
				label: "제외",
				toneClass: "pill--excluded",
				dotToken: null,
				hollowDot: true,
				spinning: false,
				title: "인덱싱에서 제외됨 · 클릭하면 포함",
			};
		case "failed":
			return {
				label: "재시도",
				toneClass: "pill--failed",
				dotToken: null,
				hollowDot: false,
				spinning: false,
				title: `인덱싱 실패 · ${reason} — 클릭하면 다시 시도`,
			};
		case "retrying":
			return {
				label: "재시도 중",
				toneClass: "pill--failed",
				dotToken: null,
				hollowDot: false,
				spinning: true,
				title: `인덱싱 실패 · ${reason} — 클릭하면 다시 시도`,
			};
	}
}

export type Permission = "ask" | "allow" | "deny";

interface PermissionMeta {
	readonly label: string;
	readonly title: string;
	readonly dotToken: string;
	readonly toneClass: string;
}

export const PERMISSIONS: Readonly<Record<Permission, PermissionMeta>> = {
	ask: { label: "Ask", title: "허용 시 공유", dotToken: "--warn-dot", toneClass: "pill--ask" },
	allow: { label: "Allow", title: "항상 허용", dotToken: "--info-dot", toneClass: "pill--allow" },
	deny: { label: "Deny", title: "항상 거부", dotToken: "--danger-dot", toneClass: "pill--deny" },
};

export function accessBadge(permission: Permission, explicit: boolean): BadgeVisual {
	const meta = PERMISSIONS[permission];
	return {
		label: meta.label,
		toneClass: explicit ? meta.toneClass : "pill--inherited",
		dotToken: meta.dotToken,
		hollowDot: false,
		spinning: false,
		title: explicit ? meta.title : `${meta.title} · 상위 폴더 설정`,
	};
}
