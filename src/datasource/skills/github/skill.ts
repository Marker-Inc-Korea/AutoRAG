/**
 * GitHub Issues/PRs datasource skill (issue #1303).
 *
 * Thin composition of GitHubConnector with the shared
 * {@link ConnectorDatasourceSkill} base: descriptor, polling metadata,
 * ok/fail indexing with path/PII-opaque diagnostics, lexical retrieval via
 * the shared pipeline, opaque `/github/<instance>/…` sources, and a
 * progressive-disclosure agent-skill manifest.
 */

import {
	ConnectorDatasourceSkill,
	type ConnectorSkillDefinition,
	type ConnectorSkillOptions,
} from "../../connector-skill.ts";
import { GitHubConnector, type GitHubConnectorOptions } from "./connector.ts";

export const GITHUB_SKILL_DEFINITION: ConnectorSkillDefinition = {
	skillName: "github",
	skillType: "github-issues",
	description: "GitHub Issues/PRs datasource",
	capabilities: ["issues", "api", "polling"],
	defaultTags: ["github", "issues"],
	contentType: "issue",
	manifestDescription:
		"Search indexed GitHub issues and pull requests for the configured repositories. Use for questions about bug reports, feature requests, and review discussions.",
	manifestNotes: ["Repository visibility is bounded by the server-configured token scopes."],
};

export interface GitHubSkillOptions extends Omit<ConnectorSkillOptions, "connector"> {
	/** Trusted connector configuration; a pre-built connector wins. */
	readonly connector?: GitHubConnector;
	readonly connectorOptions?: GitHubConnectorOptions;
}

export class GitHubSkill extends ConnectorDatasourceSkill {
	constructor(options: GitHubSkillOptions = {}) {
		const { connector, connectorOptions, ...rest } = options;
		super(GITHUB_SKILL_DEFINITION, {
			...rest,
			connector: connector ?? new GitHubConnector(connectorOptions ?? {}),
		});
	}
}
