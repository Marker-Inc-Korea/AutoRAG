import type { ReactElement } from "react";
import { Keycap } from "./primitives/Keycap";

/**
 * Status bar (handoff README §2): item and selection counts on the left, the
 * Quick Look / drag hint on the right.
 */
export function StatusBar({
	text,
	showHotkeys,
	devLabel,
}: {
	readonly text: string;
	readonly showHotkeys: boolean;
	readonly devLabel: string | null;
}): ReactElement {
	return (
		<div className="status-bar">
			<span>{text}</span>
			{devLabel === null ? null : <span className="status-bar__dev" title={devLabel}>{devLabel}</span>}
			<span className="status-bar__hint">
				{showHotkeys ? <Keycap label="space" variant="surface" /> : null}
				Quick Look · Drag into chat to ask
			</span>
		</div>
	);
}
