import { SECTIONS, sourceFileFor } from "./sidebar.mts";

export const REPO = "https://github.com/Marker-Inc-Korea/AutoRAG";
export const SITE = "https://autorag-docs.nomadamas.org";

/** GitHub URL of the Markdown source of a page. Agents read this, not the rendered HTML. */
export function rawMarkdownUrl(link: string): string {
	return `https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/docs/${sourceFileFor(link)}`;
}

/**
 * `llms.txt` for the published site: one line per page, pointing at the Markdown on GitHub. It is generated from the
 * sidebar so it can never list a page that does not exist or miss one that does.
 */
export function renderLlmsTxt(): string {
	const lines = [
		"# AutoRAG Agent",
		"",
		"> Personal librarian agent that finds, reads, and cites documents, chats, mail, and code on the user's own machine. These docs are the single source of truth for installing, indexing, connecting datasources, and driving AutoRAG from an agent.",
		"",
		`Read the Markdown source on GitHub (${REPO}/tree/main/docs), not the rendered site. Where a page and the code disagree, the code wins.`,
		"",
	];
	for (const section of SECTIONS) {
		for (const group of section.groups) {
			lines.push(`## ${section.text}: ${group.text}`, "");
			for (const item of group.items) lines.push(`- [${item.text}](${rawMarkdownUrl(item.link)})`);
			lines.push("");
		}
	}
	return `${lines.join("\n").trimEnd()}\n`;
}
