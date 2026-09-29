import type { MouseEvent, ReactElement } from "react";
import type { BadgeVisual } from "../../state/badges";
import { RotateIcon } from "../icons";
import { HollowDot, StatusDot } from "./StatusDot";

/**
 * 22px pill badge (DESIGN.md §5 PillBadge) — Index and Access variants.
 * `visual` comes from state/badges so the mapping stays pure and the tooltip
 * doubles as the accessible name.
 */
export function PillBadge({
	visual,
	onClick,
}: {
	readonly visual: BadgeVisual;
	readonly onClick?: (event: MouseEvent<HTMLButtonElement>) => void;
}): ReactElement {
	return (
		<button
			type="button"
			className={`pill ${visual.toneClass}`}
			title={visual.title}
			aria-label={visual.title}
			onClick={onClick}
		>
			{visual.spinning ? <RotateIcon className="pill__spinner" /> : null}
			{!visual.spinning && visual.hollowDot ? <HollowDot /> : null}
			{!visual.spinning && !visual.hollowDot && visual.dotToken !== null ? (
				<StatusDot token={visual.dotToken} />
			) : null}
			{visual.label}
		</button>
	);
}
