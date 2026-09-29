import type { ReactElement } from "react";

/**
 * Shortcut hint (DESIGN.md §5 Keycap). Static, never interactive: it labels a
 * shortcut, it does not invoke one. Rendered only when `showHotkeys` is on.
 */
export function Keycap({
	label,
	variant = "chrome",
}: {
	readonly label: string;
	readonly variant?: "chrome" | "surface";
}): ReactElement {
	return (
		<span className={variant === "surface" ? "keycap keycap--surface" : "keycap"} aria-hidden="true">
			{label}
		</span>
	);
}
