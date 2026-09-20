/**
 * macOS Spotlight datasource skill (issue #1350).
 *
 * Thin composition of {@link SpotlightConnector} with the shared
 * {@link ConnectorDatasourceSkill} base. The skill is **macOS only** and
 * needs no external installs — it drives the built-in `mdfind` CLI. The
 * manifest carries operator guidance for mdfind/mdls/mdutil and the Full
 * Disk Access requirement.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { SpotlightConnector, type SpotlightConnectorOptions } from "./connector.ts";

export const SPOTLIGHT_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "spotlight",
	skillType: "spotlight-search",
	description: "macOS Spotlight datasource",
	capabilities: ["files", "macos-only", "local-search"],
	defaultTags: ["spotlight", "files"],
	contentType: "document",
	manifestDescription:
		"Search files indexed by macOS Spotlight via the built-in mdfind CLI. macOS only; no external installs. Use for questions about local documents on the Mac running AutoRAG.",
	manifestNotes: [
		"## Platform requirements",
		"- macOS only: this datasource is unavailable on any other platform.",
		"- Spotlight indexing must be enabled (`sudo mdutil -i on /`; check with `mdutil -s /`).",
		"- **Full Disk Access**: Spotlight can find protected files (Mail, Messages, Safari data, parts of ~/Library), but reading their content fails unless the terminal or app running AutoRAG has Full Disk Access. Grant it under System Settings -> Privacy & Security -> Full Disk Access, add the host app (e.g. Terminal, iTerm, or the AutoRAG binary), then restart the app. Permission failures surface as `datasource-permission-denied`.",
		"",
		"## Operator cheat sheet",
		"- `mdfind <query>` — Spotlight search; `mdfind -onlyin <dir> <query>` scopes to a directory; `mdfind -name <text>` matches file names.",
		"- `mdls <file>` — show Spotlight metadata attributes (kMDItemDisplayName, kMDItemContentType, …) for one file.",
		"- `mdutil -s /` — index status; `sudo mdutil -i on /` / `-i off /` — enable/disable indexing; `sudo mdutil -E /` — erase and rebuild the index.",
		"",
		"Queries are trusted server configuration; indexing re-runs the configured mdfind queries and hydrates text content. Result metadata carries the real absolute file path (`path`) so hits are traceable back to disk; if paths must not leave this machine, run AutoRAG with a local LLM.",
	],
	nativeCliNote:
		'Spotlight\'s own `mdfind` CLI can be run directly through `bash` for ad-hoc queries (e.g. `mdfind "<query>"` or `mdfind -onlyin <dir> "<query>"`); the dedicated tool searches the configured, hydrated index.',
};

export interface SpotlightSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: SpotlightConnector;
	readonly connectorOptions?: SpotlightConnectorOptions;
}

export class SpotlightSkill extends ConnectorDatasourceSkill {
	constructor(options: SpotlightSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(SPOTLIGHT_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new SpotlightConnector(connectorOptions ?? {}),
		});
	}
}
