/**
 * Context for the verification phase: files whose NAMES matched what the
 * question describes, and the directories they sit in. These are leads the
 * agent may open first, not evidence: the name index never read a file, so
 * nothing here is registered in the evidence ledger or can be cited.
 */

/** Files listed in the context; the rest are counted, never silently dropped. */
export const MAX_LEAD_FILES = 30;
/** Directories listed in the context. */
export const MAX_LEAD_DIRECTORIES = 8;

export interface FileNameLeadHit {
	readonly entry: { readonly path: string };
	/** Every searched name that found this path. */
	readonly candidates: readonly string[];
}

export interface FileNameLeadsInput {
	/** The names that were searched, in order. */
	readonly names: readonly string[];
	/** Merged hits, best-supported first. */
	readonly hits: readonly FileNameLeadHit[];
	/** Hits before the cap, when the caller already cut the list. */
	readonly total: number;
	/** Names whose search failed, with why; reported so a gap is not read as "no such file". */
	readonly failures?: readonly string[];
}

/** Parent directory of a POSIX or Windows path, or undefined for a bare name. */
function directoryOf(path: string): string | undefined {
	const cut = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
	if (cut <= 0) return undefined;
	return path.slice(0, cut);
}

export function formatFileNameLeads(input: FileNameLeadsInput): string {
	const searched = input.names.join(", ");
	const header = `File-name leads (searched file and folder names for: ${searched}). These are matches on names only, not their contents, so treat them as unverified:`;
	const failures = (input.failures ?? []).map((line) => `\n- ${line}`).join("");
	if (input.hits.length === 0) {
		return `${header}\nNo file or folder name matched.${failures}`;
	}
	const shown = input.hits.slice(0, MAX_LEAD_FILES);
	const files = shown.map((hit) => `- ${hit.entry.path} (found by: ${hit.candidates.join(", ")})`).join("\n");
	const more = input.total > shown.length ? `\n- ${input.total - shown.length} more matched and are not listed.` : "";

	const counts = new Map<string, number>();
	for (const hit of input.hits) {
		const directory = directoryOf(hit.entry.path);
		if (directory !== undefined) counts.set(directory, (counts.get(directory) ?? 0) + 1);
	}
	const directories = [...counts.entries()]
		.sort((a, b) => b[1] - a[1])
		.slice(0, MAX_LEAD_DIRECTORIES)
		.map(([directory, count]) => `- ${directory} (${count} ${count === 1 ? "file" : "files"})`)
		.join("\n");

	return (
		`${header}\n${files}${more}${failures}\n\n` +
		`Directories holding them, most matches first (look around here when the exact file is not among the matches):\n${directories}\n\n` +
		`Open the most plausible files first, and list those directories before searching anywhere broader.`
	);
}
