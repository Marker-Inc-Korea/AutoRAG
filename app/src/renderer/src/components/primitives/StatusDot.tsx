import type { ReactElement } from "react";

/**
 * The single carrier of state color (DESIGN.md §5 StatusDot). `token` is a
 * custom-property name from tokens.css — never a literal color.
 */
export function StatusDot({ token }: { readonly token: string }): ReactElement {
	return <span className="dot" style={{ background: `var(${token})` }} aria-hidden="true" />;
}

/** Excluded-index dot: a hollow ring instead of a fill (DESIGN.md §2.7). */
export function HollowDot(): ReactElement {
	return <span className="dot dot--hollow" aria-hidden="true" />;
}
