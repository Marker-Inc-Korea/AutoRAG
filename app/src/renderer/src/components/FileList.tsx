import type { ReactElement, RefObject } from "react";
import type { FinderEntry } from "../data/entries";
import { FileRow, type RowCallbacks } from "./FileRow";
import type { Permission } from "../../../shared/settings-contract";

/**
 * Scrolling file list (handoff README §2): the search summary, the rows, and
 * the two empty states.
 */
export function FileList({
	rows,
	listRef,
	searching,
	summary,
	emptyText,
	selectedKeys,
	focusedZone,
	flashPath,
	renamePath,
	renameDraft,
	indexOverrides,
	permissions,
	callbacks,
}: {
	readonly rows: readonly FinderEntry[];
	readonly listRef: RefObject<HTMLDivElement | null>;
	readonly searching: boolean;
	readonly summary: string;
	readonly emptyText: string;
	readonly selectedKeys: readonly string[];
	readonly focusedZone: boolean;
	readonly flashPath: string | null;
	readonly renamePath: string | null;
	readonly renameDraft: string;
	readonly indexOverrides: Readonly<Record<string, boolean>>;
	readonly permissions: Readonly<Record<string, Permission>>;
	readonly callbacks: RowCallbacks;
}): ReactElement {
	return (
		<div className="list" ref={listRef}>
			{searching ? <div className="list__summary">{summary}</div> : null}
			<div role="grid" aria-label="Files">
				{rows.map((entry) => (
					<FileRow
						key={entry.path}
						entry={entry}
						selected={selectedKeys.includes(entry.path)}
						zoneFocused={focusedZone}
						flash={flashPath === entry.path}
						searching={searching}
						indexIncluded={indexOverrides[entry.path] ?? true}
						permission={permissions[entry.path] ?? "ask"}
						renameDraft={renamePath === entry.path ? renameDraft : null}
						callbacks={callbacks}
					/>
				))}
			</div>
			{rows.length === 0 ? <div className="list__empty">{emptyText}</div> : null}
		</div>
	);
}
