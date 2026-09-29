import type { ReactElement } from "react";
import type { SortKey, SortState } from "../state/sort";
import { ChevronUpIcon } from "./icons";

interface ColumnSpec {
	readonly key: SortKey;
	readonly label: string;
	readonly right: boolean;
}

const FIXED_COLUMNS: readonly ColumnSpec[] = [
	{ key: "name", label: "Name", right: false },
	{ key: "date", label: "Date Modified", right: false },
	{ key: "size", label: "Size", right: true },
];

/**
 * Column header (handoff README §2): four sort buttons on the row grid; the
 * active column is ink-colored and shows the 10px chevron, rotated when
 * descending. "Kind" becomes "Where" while searching.
 */
export function ColumnHeader({
	sort,
	searching,
	onSort,
}: {
	readonly sort: SortState | null;
	readonly searching: boolean;
	readonly onSort: (key: SortKey) => void;
}): ReactElement {
	const columns: readonly ColumnSpec[] = [
		...FIXED_COLUMNS,
		searching ? { key: "where", label: "Where", right: false } : { key: "kind", label: "Kind", right: false },
	];

	return (
		<div className="col-header">
			{columns.map((column) => {
				const active = sort !== null && sort.key === column.key;
				return (
					<button
						key={column.key}
						type="button"
						className={`col-header__sort${active ? " col-header__sort--active" : ""}${
							column.right ? " col-header__sort--right" : ""
						}`}
						title={`Sort by ${column.label}`}
						onClick={() => onSort(column.key)}
					>
						<span className="col-header__label">{column.label}</span>
						{active ? (
							<ChevronUpIcon
								className={`col-header__chevron${sort.dir < 0 ? " col-header__chevron--desc" : ""}`}
							/>
						) : null}
					</button>
				);
			})}
			<span>Index</span>
			<span>Access</span>
		</div>
	);
}
