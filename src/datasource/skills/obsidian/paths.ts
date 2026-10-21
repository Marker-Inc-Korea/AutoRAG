import { join } from "node:path";
import { sanitizeIdSegment } from "../../connector.ts";

/**
 * Path helpers for the Obsidian datasource.
 *
 * Everything AutoRAG owns for one vault lives under
 * `<workspaceRoot>/.autorag/datasources/obsidian/<instance>/`: a MinSync workspace of its own
 * (`minsync/`, never shared with the local-file index) and the staging record (`staging.json`)
 * that maps staged notes back to the vault. Callers see slash-hierarchical opaque sources of
 * the form `/obsidian/<instance>/<vault-relative note path>`.
 */

export const OBSIDIAN_DATASOURCE_ID = "obsidian";
export const OBSIDIAN_SKILL_TYPE = "obsidian-vault";

export function obsidianDatasourceDir(workspaceRoot: string, instanceId: string): string {
	return join(workspaceRoot, ".autorag", "datasources", OBSIDIAN_DATASOURCE_ID, sanitizeIdSegment(instanceId));
}

/** The MinSync workspace this vault is indexed in; staged notes live in its `files/` folder. */
export function obsidianMinSyncWorkspace(workspaceRoot: string, instanceId: string): string {
	return join(obsidianDatasourceDir(workspaceRoot, instanceId), "minsync");
}

/** Kept beside the MinSync workspace, outside the tree MinSync indexes. */
export function obsidianStagingStatePath(workspaceRoot: string, instanceId: string): string {
	return join(obsidianDatasourceDir(workspaceRoot, instanceId), "staging.json");
}

/**
 * Escape what would change the meaning of a source segment: `%` (the escape itself), `/` (the
 * separator, possible in an instance id), `#` (datasource sources never carry a fragment), `\`
 * (not valid in a virtual path) and `*` (a scope glob). Everything else, spaces and non-Latin
 * letters included, stays readable.
 */
export function encodeSourceSegment(segment: string): string {
	return segment.replace(
		/[%/#\\*]/gu,
		(character) => `%${character.charCodeAt(0).toString(16).toUpperCase().padStart(2, "0")}`,
	);
}

/** The root source of one vault: the scope that selects every note in it. */
export function obsidianInstanceSource(instanceId: string): string {
	return `/${OBSIDIAN_DATASOURCE_ID}/${encodeSourceSegment(instanceId)}`;
}

/** The source of one note, from its vault-relative path written with `/` separators. */
export function obsidianNoteSource(instanceId: string, relativePath: string): string {
	const segments = relativePath.normalize("NFC").split("/").map(encodeSourceSegment);
	return `${obsidianInstanceSource(instanceId)}/${segments.join("/")}`;
}
