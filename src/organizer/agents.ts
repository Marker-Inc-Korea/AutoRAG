import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseFrontmatter } from "@earendil-works/pi-coding-agent";

export type AgentSource = "project" | "bundled";

export interface AgentConfig {
	name: string;
	description: string;
	systemPrompt: string;
	source: AgentSource;
	filePath: string;
}

/** Directory holding the bundled agent definitions shipped with AutoRAG. */
const BUNDLED_AGENTS_DIR = join(import.meta.dirname, "agents");

function loadAgentsFromDir(dir: string, source: AgentSource): AgentConfig[] {
	if (!existsSync(dir)) return [];
	let entries: string[];
	try {
		entries = readdirSync(dir);
	} catch {
		return [];
	}
	const agents: AgentConfig[] = [];
	for (const name of entries) {
		if (!name.endsWith(".md")) continue;
		const filePath = join(dir, name);
		let content: string;
		try {
			content = readFileSync(filePath, "utf8");
		} catch {
			continue;
		}
		const { frontmatter, body } = parseFrontmatter<Record<string, string>>(content);
		if (!frontmatter.name || !frontmatter.description) continue;
		agents.push({
			name: frontmatter.name,
			description: frontmatter.description,
			systemPrompt: body,
			source,
			filePath,
		});
	}
	return agents;
}

/**
 * Discover organizer-style agent definitions (frontmatter markdown). Project
 * definitions under `<cwd>/.autorag/agents` override bundled ones by name.
 */
export function discoverAgents(cwd: string): AgentConfig[] {
	const project = loadAgentsFromDir(join(cwd, ".autorag", "agents"), "project");
	const bundled = loadAgentsFromDir(BUNDLED_AGENTS_DIR, "bundled");
	const byName = new Map<string, AgentConfig>();
	for (const agent of bundled) byName.set(agent.name, agent);
	for (const agent of project) byName.set(agent.name, agent); // project wins
	return Array.from(byName.values());
}

export function findAgent(cwd: string, name: string): AgentConfig | undefined {
	return discoverAgents(cwd).find((agent) => agent.name === name);
}
