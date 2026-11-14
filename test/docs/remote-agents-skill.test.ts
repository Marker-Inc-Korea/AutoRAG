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
