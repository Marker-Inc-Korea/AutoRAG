import type { ReactElement, RefObject } from "react";
import { FileRow, type RowCallbacks } from "./FileRow";
import type { Permission } from "../../../shared/settings-contract";
import type { StackRow } from "../state/version-family";

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
	listPath,
	callbacks,
}: {
	readonly rows: readonly StackRow[];
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
	readonly listPath: string;
	readonly callbacks: RowCallbacks;
}): ReactElement {
	return (
		<div className="list" ref={listRef}>
			{searching ? <div className="list__summary">{summary}</div> : null}
			<div role="grid" aria-label="Files">
				{rows.map((row) => (
					<FileRow
						key={row.entry.path}
						entry={row.entry}
						selected={selectedKeys.includes(row.entry.path)}
						zoneFocused={focusedZone}
						flash={flashPath === row.entry.path}
						searching={searching}
						indexIncluded={indexOverrides[row.entry.path] ?? true}
						permission={permissions[row.entry.path] ?? "ask"}
						renameDraft={renamePath === row.entry.path ? renameDraft : null}
						stackCount={row.stackCount}
						stackOpen={row.stackOpen}
						child={row.child}
						relation={row.relation}
						listPath={listPath}
						callbacks={callbacks}
					/>
				))}
			</div>
			{rows.length === 0 ? <div className="list__empty">{emptyText}</div> : null}
		</div>
	);
}
