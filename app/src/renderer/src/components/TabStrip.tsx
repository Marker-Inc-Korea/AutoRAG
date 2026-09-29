import type { MouseEvent, ReactElement } from "react";
import { CloseIcon, PlusIcon } from "./icons";
import { StatusDot } from "./primitives/StatusDot";

export interface TabView {
	readonly id: number;
	readonly title: string;
	readonly active: boolean;
	readonly flash: boolean;
}

/**
 * Tab strip (handoff README §2): 48px chrome bar with bottom-aligned 36px
 * tabs; the active tab is white and merges into the toolbar via -1px margin.
 */
export function TabStrip({
	tabs,
	canClose,
	onSelect,
	onClose,
	onNewTab,
}: {
	readonly tabs: readonly TabView[];
	readonly canClose: boolean;
	readonly onSelect: (id: number) => void;
	readonly onClose: (id: number) => void;
	readonly onNewTab: () => void;
}): ReactElement {
	return (
		<div className="tabs">
			{tabs.map((tab) => (
				<div key={tab.id} className={`tab${tab.active ? " tab--active" : ""}`}>
					{tab.flash ? <StatusDot token="--accent" /> : null}
					<button type="button" className="tab__title" onClick={() => onSelect(tab.id)}>
						{tab.title}
					</button>
					{canClose ? (
						<button
							type="button"
							className="tab__close"
							title="Close tab ⌘W"
							aria-label="Close tab ⌘W"
							onClick={(event: MouseEvent<HTMLButtonElement>) => {
								event.stopPropagation();
								onClose(tab.id);
							}}
						>
							<CloseIcon />
						</button>
					) : null}
				</div>
			))}
			<button type="button" className="tabs__new" title="New tab ⌘T" aria-label="New tab ⌘T" onClick={onNewTab}>
				<PlusIcon />
			</button>
		</div>
	);
}
