import type { ReactElement } from "react";
import { Keycap } from "./primitives/Keycap";

/**
 * Status bar (handoff README §2): item and selection counts on the left, the
 * Quick Look / drag hint on the right.
 */
export function StatusBar({
	text,
	showHotkeys,
}: {
	readonly text: string;
	readonly showHotkeys: boolean;
}): ReactElement {
	return (
		<div className="status-bar">
			<span>{text}</span>
			<span className="status-bar__hint">
				{showHotkeys ? <Keycap label="space" variant="surface" /> : null}
				Quick Look · Drag into chat to ask
			</span>
		</div>
	);
}
