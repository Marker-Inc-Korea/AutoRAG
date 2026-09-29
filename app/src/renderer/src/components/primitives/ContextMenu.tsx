import { useEffect, useRef } from "react";
import type { ReactElement } from "react";
import {
	CONTEXT_MENU_WIDTH,
	estimateMenuHeight,
	type FinderMenuAction,
	type FinderMenuEntry,
	menuPlacement,
} from "../../state/context-menu";
import { Keycap } from "./Keycap";

/**
 * Row context menu (DESIGN.md §5 ContextMenu, row variant): fixed at the
 * cursor, 220px, closes on outside mousedown. Esc is handled by the window
 * keymap so the whole dismissal order stays in one place.
 */
export function ContextMenu({
	cursor,
	entries,
	onSelect,
	onClose,
}: {
	readonly cursor: { readonly x: number; readonly y: number };
	readonly entries: readonly FinderMenuEntry[];
	readonly onSelect: (action: FinderMenuAction) => void;
	readonly onClose: () => void;
}): ReactElement {
	const ref = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const onMouseDown = (event: globalThis.MouseEvent): void => {
			const node = ref.current;
			if (node !== null && event.target instanceof Node && node.contains(event.target)) {
				return;
			}
			onClose();
		};
		window.addEventListener("mousedown", onMouseDown);
		return () => window.removeEventListener("mousedown", onMouseDown);
	}, [onClose]);

	const placement = menuPlacement(
		cursor,
		{ width: CONTEXT_MENU_WIDTH, height: estimateMenuHeight(entries) },
		{ width: window.innerWidth, height: window.innerHeight },
	);

	return (
		<div
			ref={ref}
			className="menu"
			role="menu"
			aria-label="파일 작업"
			style={{ left: `${placement.left}px`, top: `${placement.top}px` }}
		>
			{entries.map((entry, index) =>
				entry.kind === "separator" ? (
					// biome-ignore lint/suspicious/noArrayIndexKey: separators carry no identity
					<div key={`sep-${index}`} className="menu__separator" role="separator" />
				) : (
					<button
						key={entry.action}
						type="button"
						role="menuitem"
						className={`menu__item${entry.tone === "danger" ? " menu__item--danger" : ""}${
							entry.tone === "warning" ? " menu__item--warning" : ""
						}`}
						disabled={entry.disabled}
						onClick={() => onSelect(entry.action)}
					>
						<span className="menu__label">{entry.label}</span>
						{entry.keycap === null ? null : <Keycap label={entry.keycap} />}
					</button>
				),
			)}
		</div>
	);
}
