/**
 * Inline icon set — Lucide-style strokes on a 24 × 24 viewBox, stroke-width
 * 1.8–2.6, round caps (handoff README "Assets"). No icon package, no emoji.
 *
 * Path data is the reference prototype's, verbatim.
 */

import type { ReactElement } from "react";

interface GlyphProps {
	readonly size?: number | undefined;
	readonly className?: string | undefined;
}

interface StrokeProps extends GlyphProps {
	readonly width?: number | undefined;
	readonly children: ReactElement | readonly ReactElement[];
}

function Stroke({ size = 16, width = 1.8, className, children }: StrokeProps): ReactElement {
	return (
		<svg
			className={className}
			width={size}
			height={size}
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth={width}
			strokeLinecap="round"
			strokeLinejoin="round"
			aria-hidden="true"
			focusable="false"
		>
			{children}
		</svg>
	);
}

/** Sidebar place icons carry their path in the nav data. */
export function PathIcon({ d, size = 16, className }: GlyphProps & { readonly d: string }): ReactElement {
	return (
		<Stroke size={size} className={className}>
			<path d={d} />
		</Stroke>
	);
}

export function BellIcon(props: GlyphProps): ReactElement {
	return (
		<Stroke {...props}>
			<path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
			<path d="M10.3 21a1.9 1.9 0 0 0 3.4 0" />
		</Stroke>
	);
}

export function GearIcon(props: GlyphProps): ReactElement {
	return (
		<Stroke {...props}>
			<circle cx="12" cy="12" r="3" />
			<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
		</Stroke>
	);
}

export function SearchIcon({ size = 14, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.2} className={className}>
			<circle cx="11" cy="11" r="7" />
			<path d="M20 20l-3.5-3.5" />
		</Stroke>
	);
}

export function ChevronLeftIcon({ size = 16, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.2} className={className}>
			<path d="M15 6l-6 6 6 6" />
		</Stroke>
	);
}

export function ChevronRightIcon({ size = 16, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.2} className={className}>
			<path d="M9 6l6 6-6 6" />
		</Stroke>
	);
}

export function ChevronUpIcon({ size = 10, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={3} className={className}>
			<path d="M6 15l6-6 6 6" />
		</Stroke>
	);
}

export function CloseIcon({ size = 10, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={3} className={className}>
			<path d="M6 6l12 12M18 6L6 18" />
		</Stroke>
	);
}

export function PlusIcon({ size = 14, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.4} className={className}>
			<path d="M12 5v14M5 12h14" />
		</Stroke>
	);
}

export function SendIcon({ size = 16, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.4} className={className}>
			<path d="M4 4l16 8-16 8 3-8z" />
			<path d="M7 12h13" />
		</Stroke>
	);
}

export function SparklesIcon({ size = 14, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2} className={className}>
			<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z" />
		</Stroke>
	);
}

export function ArrowUpIcon({ size = 16, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.6} className={className}>
			<path d="M12 19V5M6 11l6-6 6 6" />
		</Stroke>
	);
}

export function RotateIcon({ size = 10, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2.8} className={className}>
			<path d="M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5" />
		</Stroke>
	);
}

export function HistoryIcon({ size = 16, className }: GlyphProps): ReactElement {
	return (
		<Stroke size={size} width={2} className={className}>
			<path d="M3 12a9 9 0 1 0 3-6.7L3 8" />
			<path d="M3 3v5h5" />
			<path d="M12 7v5l3 2" />
		</Stroke>
	);
}

/** Folders render the filled glyph in --folder, never a tile (DESIGN.md §5). */
export function FolderGlyph({ size = 18 }: GlyphProps): ReactElement {
	return (
		<svg
			className="tile tile--folder"
			width={size}
			height={size}
			viewBox="0 0 24 24"
			aria-hidden="true"
			focusable="false"
		>
			<path
				d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2h8.5A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z"
				fill="currentColor"
			/>
		</svg>
	);
}
