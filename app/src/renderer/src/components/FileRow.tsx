import type { ChangeEvent, KeyboardEvent, MouseEvent, ReactElement } from "react";
import { accessBadge, indexBadge } from "../state/badges";
import type { Permission } from "../../../shared/settings-contract";
import type { FinderEntry } from "../data/entries";
import { whereSegments } from "../state/paths";
import { relationLabel, type VersionRelation } from "../state/version-family";
import { ChevronRightIcon } from "./icons";
import { FileTile } from "./primitives/FileTile";
import { PillBadge } from "./primitives/PillBadge";

export interface RowCallbacks {
	readonly onClick: (entry: FinderEntry, event: MouseEvent<HTMLDivElement>) => void;
	readonly onOpen: (entry: FinderEntry) => void;
	readonly onContextMenu: (entry: FinderEntry, event: MouseEvent<HTMLDivElement>) => void;
	readonly onToggleIndex: (entry: FinderEntry) => void;
	readonly onToggleStack: (entry: FinderEntry) => void;
	readonly onChangeAccess: (entry: FinderEntry) => void;
	readonly onRenameChange: (value: string) => void;
	readonly onRenameCommit: () => void;
	readonly onRenameCancel: () => void;
}

/**
 * One 32px file row on --grid-file-row (handoff README §2 "File rows").
 * Selection is a wash, never an edge: --accent-soft while the Finder zone owns
 * focus, --bg-input otherwise.
 */
export function FileRow({
	entry,
	selected,
	zoneFocused,
	flash,
	searching,
	indexIncluded,
	permission,
	renameDraft,
	stackCount,
	stackOpen,
	child,
	relation,
	listPath,
	callbacks,
}: {
	readonly entry: FinderEntry;
	readonly selected: boolean;
	readonly zoneFocused: boolean;
	readonly flash: boolean;
	readonly searching: boolean;
	readonly indexIncluded: boolean;
	readonly permission: Permission;
	readonly renameDraft: string | null;
	readonly stackCount: number;
	readonly stackOpen: boolean;
	readonly child: boolean;
	readonly relation: VersionRelation | null;
	readonly listPath: string;
	readonly callbacks: RowCallbacks;
}): ReactElement {
	const classes = ["row"];
	if (selected) {
		classes.push(zoneFocused ? "row--selected-focused" : "row--selected");
	}
	if (flash) {
		classes.push("row--flash");
	}
	if (child) {
		classes.push("row--child");
	}

	return (
		<div
			className={classes.join(" ")}
			role="row"
			aria-selected={selected}
			onClick={(event) => callbacks.onClick(entry, event)}
			onDoubleClick={() => callbacks.onOpen(entry)}
			onContextMenu={(event) => callbacks.onContextMenu(entry, event)}
		>
			<div className="row__name" role="gridcell">
				{child ? <span className="row__connector" aria-hidden="true" /> : <FileTile kind={entry.fileKind} />}
				{renameDraft === null ? (
					<span className="row__label">{entry.name}</span>
				) : (
					<input
						// biome-ignore lint/a11y/noAutofocus: inline rename must take the caret immediately
						autoFocus
						className="row__rename"
						type="text"
						value={renameDraft}
						aria-label="새 이름"
						onChange={(event: ChangeEvent<HTMLInputElement>) => callbacks.onRenameChange(event.target.value)}
						onClick={(event: MouseEvent<HTMLInputElement>) => event.stopPropagation()}
						onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
							if (event.key === "Enter") {
								event.preventDefault();
								callbacks.onRenameCommit();
							}
							if (event.key === "Escape") {
								event.preventDefault();
								callbacks.onRenameCancel();
							}
						}}
						onBlur={callbacks.onRenameCancel}
					/>
				)}
				{stackCount > 0 ? (
					<button
						type="button"
						className={stackOpen ? "row__stackbadge row__stackbadge--open" : "row__stackbadge"}
						title={`다른 버전 ${stackCount}개 · 클릭하면 펼쳐집니다`}
						aria-label={`다른 버전 ${stackCount}개 · 클릭하면 펼쳐집니다`}
						aria-expanded={stackOpen}
						onClick={(event) => {
							event.stopPropagation();
							callbacks.onToggleStack(entry);
						}}
					>
						<ChevronRightIcon size={9} />
						{stackCount}
					</button>
				) : null}
			</div>
			<span className="row__meta" role="gridcell">
				{entry.dateLabel}
			</span>
			<span className="row__meta row__meta--right" role="gridcell">
				{entry.sizeLabel}
			</span>
			<span className="row__meta" role="gridcell">
				{child && relation !== null
					? relationLabel(relation, entry.location, listPath)
					: searching
						? whereSegments(entry.location)
						: entry.kindLabel}
			</span>
			<div className="row__cell" role="gridcell">
				{entry.kind === "file" ? (
					<PillBadge
						visual={indexBadge(indexIncluded ? "included" : "excluded")}
						onClick={(event) => {
							event.stopPropagation();
							callbacks.onToggleIndex(entry);
						}}
					/>
				) : null}
			</div>
			<div className="row__cell" role="gridcell">
				<PillBadge
					visual={accessBadge(permission, false)}
					onClick={(event) => {
						event.stopPropagation();
						callbacks.onChangeAccess(entry);
					}}
				/>
			</div>
		</div>
	);
}
