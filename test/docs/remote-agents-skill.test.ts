import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { registeredMcpTools } from "../helpers/mcp-tool-names.ts";

const SKILL = readFileSync(new URL("../../plugins/remote-agents/skills/search/SKILL.md", import.meta.url), "utf8");

function frontmatter(text: string): Record<string, string> {
	const match = /^---\n([\s\S]*?)\n---\n/.exec(text);
	if (match?.[1] === undefined) throw new Error("SKILL.md has no frontmatter");
	return Object.fromEntries(
		match[1].split("\n").flatMap((line) => {
			const at = line.indexOf(":");
			return at < 0 ? [] : [[line.slice(0, at).trim(), line.slice(at + 1).trim()]];
		}),
	);
}

describe("remote-agents search skill", () => {
	it("has the Agent Skills frontmatter: a kebab-case name matching its directory and a description with a trigger", () => {
		const meta = frontmatter(SKILL);
		expect(meta.name).toBe("search");
		expect(meta.description?.length ?? 0).toBeGreaterThan(40);
		expect(meta.description?.length ?? 0).toBeLessThanOrEqual(1024);
		expect(meta.description).toMatch(/use when/i);
	});

	it("names only tools the server really registers, in both modes", async () => {
		const registered = (await registeredMcpTools({ readOnly: false })).map((tool) => tool.name);
		const mentioned = [...new Set(SKILL.match(/\bautorag\.[a-z]+(?:\.[a-z]+)?\b/g) ?? [])];
		expect(mentioned.length).toBeGreaterThan(0);
		for (const name of mentioned) expect(registered, name).toContain(name);
	});

	it("only passes search arguments the tool's schema defines", async () => {
		const search = (await registeredMcpTools({ readOnly: true })).find((tool) => tool.name === "autorag.search");
		const properties = Object.keys(search?.inputSchema.properties ?? {});
		expect(properties.length).toBeGreaterThan(0);
		const named = [...SKILL.matchAll(/`(topK|scope|strict|datasourceIds|methods|local|query)`/g)].map(
			(m) => m[1] ?? "",
		);
		expect(named.length).toBeGreaterThan(0);
		for (const argument of new Set(named)) expect(properties, argument).toContain(argument);
	});

	it("tells the agent the server is read-only and never to refresh, report or convert on its own", async () => {
		const readOnly = (await registeredMcpTools({ readOnly: true })).map((tool) => tool.name);
		expect(readOnly).not.toContain("autorag.refresh");
		expect(SKILL).toMatch(/read-only/i);
		expect(SKILL).toMatch(/autorag\.refresh/);
		expect(SKILL).toMatch(/not (?:offered|available)|absent|missing/i);
	});

	it("treats retrieved text as data and keeps private archives out of other tools", () => {
		expect(SKILL).toMatch(/instructions? (?:inside|in|within) (?:a |any )?(?:document|message|email|result)/i);
		expect(SKILL).toMatch(/do not (?:forward|send|share)/i);
	});

	it("asks for plain-text source attribution, with no vendor-specific citation markup", () => {
		expect(SKILL).toMatch(/file name|source/i);
		expect(SKILL).not.toMatch(/mcp__autorag__/);
	});

	it("explains how a host prefixes the tool names, instead of assuming one host's spelling", () => {
		expect(SKILL).toMatch(/prefix/i);
	});
});

const ROOT = new URL("../../plugins/remote-agents/", import.meta.url);

function readJson(relative: string): Record<string, unknown> {
	const parsed: unknown = JSON.parse(readFileSync(new URL(relative, ROOT), "utf8"));
	if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed))
		throw new Error(`${relative} is not an object`);
	return Object.fromEntries(Object.entries(parsed));
}

describe("remote-agents portable manifest", () => {
	it("is an Agent Plugins v1 manifest with a kebab-case name and semver version", () => {
		const manifest = readJson("plugin.json");
		expect(manifest.$schema).toBe("https://agent-plugins.org/schemas/1.0.0/plugin.schema.json");
		expect(manifest.name).toMatch(/^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/);
		expect(manifest.version).toMatch(/^\d+\.\d+\.\d+$/);
		expect(String(manifest.description).length).toBeGreaterThan(40);
		expect(manifest.license).toBe("Apache-2.0");
	});

	it("declares no MCP server: a portable manifest cannot carry a per-user URL, so the user registers theirs", () => {
		const names = readdirSync(fileURLToPath(ROOT));
		expect(names).not.toContain("mcp.json");
		expect(readJson("plugin.json")).not.toHaveProperty("mcpServers");
	});

	it("does not collide with the Hermes package's plugin name", () => {
		const hermes = JSON.parse(readFileSync(new URL("../../plugins/hermes/plugin.json", import.meta.url), "utf8"));
		expect(readJson("plugin.json").name).not.toBe(hermes.name);
	});

	it("keeps the Claude Code manifest (which cannot read the portable one) identical in every shared field", () => {
		const portable = readJson("plugin.json");
		const claude = readJson(".claude-plugin/plugin.json");
		for (const [key, value] of Object.entries(claude)) expect(portable[key], key).toEqual(value);
		expect(Object.keys(claude)).toEqual(expect.arrayContaining(["name", "version", "description"]));
	});
});

describe("repo marketplaces resolve to the shipped plugin", () => {
	const REPO = new URL("../../", import.meta.url);
	const readRepoJson = (relative: string): Record<string, unknown> => {
		const parsed: unknown = JSON.parse(readFileSync(new URL(relative, REPO), "utf8"));
		if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed))
			throw new Error(`${relative} is not an object`);
		return Object.fromEntries(Object.entries(parsed));
	};
	const entries = (marketplace: Record<string, unknown>): Record<string, unknown>[] =>
		Array.isArray(marketplace.plugins)
			? marketplace.plugins.filter((p): p is Record<string, unknown> => typeof p === "object" && p !== null)
			: [];

	it("Claude Code reads the marketplace from the repository root's .claude-plugin, which is where it looks after `marketplace add owner/repo`", () => {
		const marketplace = readRepoJson(".claude-plugin/marketplace.json");
		expect(marketplace.name).toBe("autorag");
		const [entry] = entries(marketplace);
		expect(entry?.name).toBe("autorag-remote");
		expect(entry?.source).toBe("./plugins/remote-agents");
	});

	it("Codex and Claude point at the same plugin directory and the same name", () => {
		const claude = entries(readRepoJson(".claude-plugin/marketplace.json"))[0];
		const codex = entries(readRepoJson(".agents/plugins/marketplace.json"))[0];
		expect(codex?.name).toBe(claude?.name);
		const source: unknown = codex?.source;
		expect(typeof source === "object" && source !== null && "path" in source ? source.path : source).toBe(
			claude?.source,
		);
	});

	it("the plugin name in each marketplace is the name inside the plugin's own manifest", () => {
		const manifest = readJson("plugin.json");
		for (const file of [".claude-plugin/marketplace.json", ".agents/plugins/marketplace.json"]) {
			expect(entries(readRepoJson(file))[0]?.name, file).toBe(manifest.name);
		}
	});

	it("Cursor reads a root .cursor-plugin/marketplace.json with the same plugin, a kebab-case name, an owner and no `..` path", () => {
		const marketplace = readRepoJson(".cursor-plugin/marketplace.json");
		expect(marketplace.name).toBe("autorag");
		expect(marketplace.owner).toMatchObject({ name: expect.any(String) });
		const [entry] = entries(marketplace);
		expect(entry?.name).toBe(readJson("plugin.json").name);
		expect(String(entry?.source)).not.toMatch(/\.\.|^\//);
		expect(String(entry?.source).replace(/^\.\//, "")).toBe("plugins/remote-agents");
		expect(String(entry?.description).length).toBeGreaterThan(20);
	});

	it("does not leave a second, unreachable Claude marketplace under plugins/", () => {
		expect(() => readFileSync(new URL("plugins/.claude-plugin/marketplace.json", REPO))).toThrow();
	});
});

describe("plugins/remote-agents README (the directory listing's description)", () => {
	const README = readFileSync(new URL("README.md", ROOT), "utf8");
	const prose = README.replace(/```[\s\S]*?```/g, "");

	it("has at least 40 words outside code blocks, which Claude's directory requires", () => {
		expect(prose.split(/\s+/).filter(Boolean).length).toBeGreaterThanOrEqual(40);
	});

	it("says plainly that the package runs nothing, ships no MCP server and sends nothing, and what the skill does do", () => {
		expect(README).toMatch(/skill only/i);
		expect(README).toMatch(/no (?:mcp )?server|does not (?:run|include|start)/i);
		expect(README).toMatch(/nothing (?:is )?(?:sent|uploaded|fetched)|does not (?:send|fetch|upload)/i);
	});

	it("tells the user how to connect their own server, linking the doc that exists", () => {
		expect(README).toContain("docs/remote-agents.md");
		expect(README).toMatch(/autorag-mcp --http/);
	});

	it("names the owner token as a secret the plugin never asks for", () => {
		expect(README).toMatch(/owner token/i);
		expect(README).toMatch(/never (?:ask|request|store)/i);
	});

	it("discloses that what a search returns is read by the connected agent's model, wherever that agent runs", () => {
		// "Sends nothing" is true of the package and false of the setup it enables; a reader must not stop at the first.
		expect(prose).toMatch(/(?:chunks|results|passages|excerpts)[^.]*\b(?:read|seen|processed) by/i);
		expect(prose).toMatch(/vendor|cloud|wherever that agent runs|the agent's (?:model|provider)/i);
		expect(prose).toMatch(/only (?:the )?(?:results|chunks|passages) (?:of )?(?:a |the )?search/i);
	});

	it("only names tools and flags that exist", async () => {
		const registered = await registeredMcpTools({ readOnly: false });
		const names = new Set(registered.map((tool) => tool.name));
		for (const name of new Set(README.match(/\bautorag\.[a-z]+(?:\.[a-z]+)?\b/g) ?? []))
			expect(names.has(name), name).toBe(true);
	});
});
