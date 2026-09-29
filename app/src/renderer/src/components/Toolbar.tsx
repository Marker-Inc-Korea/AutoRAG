import { Fragment } from "react";
import type { ChangeEvent, KeyboardEvent, ReactElement, RefObject } from "react";
import type { Crumb } from "../state/paths";
import { ChevronLeftIcon, ChevronRightIcon, SearchIcon } from "./icons";
import { Keycap } from "./primitives/Keycap";

/**
 * Toolbar (handoff README §2): back/forward, truncated breadcrumbs, and the
 * 236 × 32 instant-search field.
 */
export function Toolbar({
	crumbs,
	canGoBack,
	canGoForward,
	query,
	placeholder,
	searchFocused,
	showHotkeys,
	searchRef,
	onNavigate,
	onBack,
	onForward,
	onQueryChange,
	onSearchKeyDown,
	onSearchFocus,
	onSearchBlur,
}: {
	readonly crumbs: readonly Crumb[];
	readonly canGoBack: boolean;
	readonly canGoForward: boolean;
	readonly query: string;
	readonly placeholder: string;
	readonly searchFocused: boolean;
	readonly showHotkeys: boolean;
	readonly searchRef: RefObject<HTMLInputElement | null>;
	readonly onNavigate: (path: string) => void;
	readonly onBack: () => void;
	readonly onForward: () => void;
	readonly onQueryChange: (value: string) => void;
	readonly onSearchKeyDown: (event: KeyboardEvent<HTMLInputElement>) => void;
	readonly onSearchFocus: () => void;
	readonly onSearchBlur: () => void;
}): ReactElement {
	return (
		<div className="toolbar">
			<button
				type="button"
				className="icon-button"
				title="Back"
				aria-label="Back"
				disabled={!canGoBack}
				onClick={onBack}
			>
				<ChevronLeftIcon />
			</button>
			<button
				type="button"
				className="icon-button"
				title="Forward"
				aria-label="Forward"
				disabled={!canGoForward}
				onClick={onForward}
			>
				<ChevronRightIcon />
			</button>
			<nav className="crumbs" aria-label="Location">
				{crumbs.map((crumb) => (
					<Fragment key={crumb.path}>
						{crumb.separator ? <span className="crumbs__sep">/</span> : null}
						<button
							type="button"
							className={`crumb${crumb.isLast ? " crumb--last" : ""}`}
							title={crumb.label}
							onClick={() => onNavigate(crumb.path)}
						>
							{crumb.label}
						</button>
					</Fragment>
				))}
			</nav>
			<div className={`search${searchFocused ? " search--focused" : ""}`}>
				<SearchIcon className="search__icon" />
				<input
					ref={searchRef}
					className="search__input"
					type="text"
					value={query}
					placeholder={placeholder}
					aria-label={placeholder}
					onChange={(event: ChangeEvent<HTMLInputElement>) => onQueryChange(event.target.value)}
					onKeyDown={onSearchKeyDown}
					onFocus={onSearchFocus}
					onBlur={onSearchBlur}
				/>
				{showHotkeys ? <Keycap label="⌘F" /> : null}
			</div>
		</div>
	);
}
