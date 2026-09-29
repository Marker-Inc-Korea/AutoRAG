import type { ReactElement } from "react";
import { INDEXING_STATUS, NAV_GROUPS, type NavItem } from "../data/places";
import { BellIcon, GearIcon, PathIcon } from "./icons";
import { FileTile } from "./primitives/FileTile";
import { Keycap } from "./primitives/Keycap";
import { StatusDot } from "./primitives/StatusDot";

/**
 * Sidebar (handoff README §1): native window chrome is supplied by Electron,
 * so the renderer starts directly with the navigation groups.
 */
export function Sidebar({
	activeLabel,
	pendingRequests,
	showHotkeys,
	onNavigate,
	onOpenRequests,
	onOpenSettings,
}: {
	readonly activeLabel: string | null;
	readonly pendingRequests: number;
	readonly showHotkeys: boolean;
	readonly onNavigate: (item: NavItem) => void;
	readonly onOpenRequests?: (() => void) | undefined;
	readonly onOpenSettings?: (() => void) | undefined;
}): ReactElement {
	return (
		<aside className="sidebar" aria-label="Places">
			<div className="sidebar__scroll">
				{NAV_GROUPS.map((group) => (
					<div key={group.label} className="sidebar__group">
						<div className="sidebar__label">{group.label}</div>
						{group.items.map((item) => {
							const active = item.label === activeLabel;
							return (
								<button
									key={item.label}
									type="button"
									className={`sidebar__item${active ? " sidebar__item--active" : ""}`}
									aria-current={active ? "true" : undefined}
									onClick={() => onNavigate(item)}
								>
									{item.sourceKind === null ? (
										<PathIcon className="sidebar__item-icon" d={item.icon ?? ""} />
									) : (
										<FileTile kind={item.sourceKind} size={16} />
									)}
									<span className="sidebar__item-label">{item.label}</span>
									{item.syncing ? <StatusDot token="--warn-dot" /> : null}
									{item.error ? <StatusDot token="--error-orange-dot" /> : null}
								</button>
							);
						})}
					</div>
				))}
			</div>
			<div className="sidebar__footer">
				<div className="sidebar__indexing">
					<div className="sidebar__indexing-row">
						<span>{INDEXING_STATUS.label}</span>
						<span className="sidebar__indexing-value">{INDEXING_STATUS.percent}%</span>
					</div>
					<div className="sidebar__progress">
						<div className="sidebar__progress-fill" style={{ width: `${INDEXING_STATUS.percent}%` }} />
					</div>
				</div>
				<button type="button" className="sidebar__item" onClick={onOpenRequests}>
					<BellIcon className="sidebar__item-icon" />
					<span className="sidebar__item-label">Requests</span>
					{pendingRequests > 0 ? <span className="sidebar__count">{pendingRequests}</span> : null}
				</button>
				<button type="button" className="sidebar__item" onClick={onOpenSettings}>
					<GearIcon className="sidebar__item-icon" />
					<span className="sidebar__item-label">Settings</span>
					{showHotkeys ? <Keycap label="⌘," /> : null}
				</button>
			</div>
		</aside>
	);
}
