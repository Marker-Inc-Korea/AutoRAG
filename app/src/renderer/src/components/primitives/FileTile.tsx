import type { ReactElement } from "react";
import { type FileKind, kindMeta } from "../../state/kinds";
import { FolderGlyph } from "../icons";

/**
 * File-type tile (DESIGN.md §5 FileTile): rounded square, white letter, color
 * from the KD map. Folders render the folder glyph instead.
 */
export function FileTile({
	kind,
	size = 18,
}: {
	readonly kind: FileKind;
	readonly size?: 16 | 18 | 22;
}): ReactElement {
	if (kind === "folder") {
		return <FolderGlyph size={size} />;
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
