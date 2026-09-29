/**
 * Dotfile visibility for the Finder list.
 *
 * macOS Finder semantics: names starting with "." are hidden by default.
 * When the "Show hidden files" setting is on they stay in the list but are
 * rendered dimmed (see `.row--hidden`), so hidden files are recognizable.
 */

export function isHiddenName(name: string): boolean {
	return name.startsWith(".");
}

export function visibleEntries<T extends { readonly name: string }>(
	entries: readonly T[],
	showHidden: boolean,
): T[] {
	return showHidden ? [...entries] : entries.filter((entry) => !isHiddenName(entry.name));
}
