import type { ReactElement } from "react";
import { type FileKind, kindMeta } from "../../state/kinds";
import { FolderGlyph } from "../icons";

/**
 * File-type tile (DESIGN.md §5 FileTile): rounded square, white letter, color
 * from the KD map. Folders render the folder glyph instead. When the OS
 * produced a tile icon (a Finder thumbnail), it renders inside the same square.
 */
export function FileTile({
	kind,
	iconDataUrl,
	size = 18,
}: {
	readonly kind: FileKind;
	readonly iconDataUrl?: string | null;
	readonly size?: 16 | 18 | 22;
}): ReactElement {
	if (kind === "folder") {
		return <FolderGlyph size={size} />;
	}
	if (iconDataUrl !== undefined && iconDataUrl !== null) {
		return (
			<img
				className={`tile tile--${size} tile--image`}
				src={iconDataUrl}
				alt=""
				aria-hidden="true"
				draggable={false}
			/>
		);
	}
	const meta = kindMeta(kind);
	return (
		<span
			className={`tile tile--${size}`}
			style={{ background: `var(${meta.tileToken})`, color: `var(${meta.foregroundToken})` }}
			aria-hidden="true"
		>
			{meta.letter}
		</span>
	);
}
