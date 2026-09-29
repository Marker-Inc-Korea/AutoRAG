import type { ReactElement } from "react";
import { StatusDot } from "./StatusDot";

/**
 * Bottom-center toast (DESIGN.md §5 Toast). A new message replaces the current
 * one and it auto-hides after --time-toast; the live region is the only way the
 * message reaches a screen reader (DESIGN.md §8 constraint 6).
 */
export function Toast({ message }: { readonly message: string }): ReactElement {
	return (
		<output className="toast" aria-live="polite">
			<StatusDot token="--ok-dot" />
			{message}
		</output>
	);
}
