import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { parse as parseYaml } from "yaml";
import { renderLlmsTxt, SITE } from "../../docs/.vitepress/llms.mts";
import { builtinDatasourceSkillNames } from "../../src/datasource/skills/factory.ts";
import {
	allPages,
	anchorsOf,
	DOCS_ROOT,
	exists,
	fencedBlocks,
	inlineCode,
	isExternal,
	linksOf,
	REPO_ROOT,
	readDocPages,
	resolveDocLink,
	SECTIONS,
	sidebarFiles,
	stripFences,
} from "../helpers/docs-site.ts";
import { registeredMcpToolNames } from "../helpers/mcp-tool-names.ts";

const pages = readDocPages();
const byFile = new Map(pages.map((page) => [page.file, page]));
/** Pages that publish content: everything the sidebar lists. The home page and the folder README are not listed. */
const published = allPages().map((item) => ({ ...item, file: `${item.link.replace(/^\//, "")}.md` }));
const TOOLING_PAGES = new Set(["index.md", "README.md"]);

function frontmatter(text: string): { data: Record<string, unknown>; body: string } | undefined {
	const match = /^---\n([\s\S]*?)\n---\n/.exec(text);
	if (!match) return undefined;
	return { data: (parseYaml(match[1]) ?? {}) as Record<string, unknown>, body: text.slice(match[0].length) };
}

describe("docs site: table of contents", () => {
	it("lists every page exactly once, with a link under its section", () => {
		const links = allPages().map((item) => item.link);
		expect(new Set(links).size, "duplicate sidebar link").toBe(links.length);
		for (const section of SECTIONS) {
			for (const group of section.groups) {
				for (const item of group.items) {
					expect(item.link.startsWith(section.base), `${item.link} is outside ${section.base}`).toBe(true);
				}
			}
		}
	});

	it("has a Markdown file for every sidebar entry and a sidebar entry for every Markdown file", () => {
		const listed = new Set(sidebarFiles());
		for (const file of listed) expect(byFile.has(file), `sidebar entry without a page: docs/${file}`).toBe(true);
		for (const page of pages) {
			if (TOOLING_PAGES.has(page.file)) continue;
			expect(listed.has(page.file), `docs/${page.file} is not in docs/.vitepress/sidebar.mts`).toBe(true);
		}
	});

	it("keeps no flat legacy pages beside the categorised tree", () => {
		const topLevel = readdirSync(DOCS_ROOT, { withFileTypes: true })
			.filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
			.map((entry) => entry.name)
			.sort();
		expect(topLevel).toEqual(["README.md", "index.md"]);
	});
});

describe("docs site: page format", () => {
	it.each(published)(
		"$link has frontmatter, one H1 equal to its title, and sane headings",
		({ file, text: sidebarText }) => {
			const page = byFile.get(file);
			expect(page, `missing docs/${file}`).toBeDefined();
			const parsed = frontmatter(page?.text ?? "");
			expect(parsed, `docs/${file}: missing YAML frontmatter`).toBeDefined();
			const { data, body } = parsed ?? { data: {}, body: "" };

			expect(Object.keys(data).sort(), `docs/${file}: frontmatter must be exactly title + description`).toEqual([
				"description",
				"title",
			]);
			expect(data.title, `docs/${file}: title must equal the sidebar text`).toBe(sidebarText);
			const description = String(data.description ?? "");
			expect(description.length, `docs/${file}: description length`).toBeGreaterThanOrEqual(50);
			expect(description.length, `docs/${file}: description length`).toBeLessThanOrEqual(160);
			expect(description, `docs/${file}: description must not contain angle brackets`).not.toMatch(/[<>]/);

			const headings = [...stripFences(body).matchAll(/^(#{1,6})\s+(.+?)\s*$/gm)].map((m) => ({
				level: m[1].length,
				text: m[2],
			}));
			expect(
				headings.filter((h) => h.level === 1).map((h) => h.text),
				`docs/${file}: exactly one H1`,
			).toEqual([sidebarText]);
			expect(headings[0]?.level, `docs/${file}: the H1 comes first`).toBe(1);
			let previous = 1;
			for (const heading of headings.slice(1)) {
				expect(heading.level, `docs/${file}: "${heading.text}" skips a heading level`).toBeLessThanOrEqual(
					previous + 1,
				);
				previous = heading.level;
			}
			const texts = headings.map((h) => h.text);
			expect(new Set(texts).size, `docs/${file}: duplicate heading text makes anchors unstable`).toBe(texts.length);
			expect(stripFences(body).replace(/^#.*$/m, "").trim().length, `docs/${file}: has no content`).toBeGreaterThan(
				200,
			);
		},
	);

	it("labels every fenced code block with a language", () => {
		for (const page of pages) {
			const fences = [...page.text.matchAll(/^(```+|~~~+)([^\n]*)$/gm)];
			// Fences come in open/close pairs; only the opening one carries the language.
			for (let i = 0; i < fences.length; i += 2) {
				expect(fences[i][2].trim(), `docs/${page.file}: unlabeled code fence`).not.toBe("");
			}
		}
	});
});

describe("docs site: links", () => {
	it("resolves every relative link and anchor, and keeps repo files behind absolute GitHub URLs", () => {
		for (const page of pages) {
			const here = anchorsOf(page.text);
			for (const link of linksOf(page.text)) {
				const where = `docs/${page.file}: ${link.raw}`;
				if (isExternal(link.target)) continue;
				if (link.target === "") {
					expect(here.has(link.hash), `${where}: no such heading on this page`).toBe(true);
					continue;
				}
				const absolute = resolveDocLink(page.file, link.target);
				expect(absolute.startsWith(DOCS_ROOT), `${where}: leaves docs/; use an absolute GitHub URL`).toBe(true);
				expect(exists(absolute), `${where}: target does not exist`).toBe(true);
				if (link.target.endsWith(".md") || link.hash !== "") {
					expect(link.target.endsWith(".md"), `${where}: link pages with the .md extension`).toBe(true);
				}
				if (link.hash !== "") {
					const target = readFileSync(absolute, "utf8").replace(/\r\n?/g, "\n");
					expect(anchorsOf(target).has(link.hash), `${where}: no such heading in the target`).toBe(true);
				}
			}
		}
	});

	it("points GitHub blob links at files that exist", () => {
		const blob = /https:\/\/github\.com\/Marker-Inc-Korea\/AutoRAG\/(?:blob|tree)\/main\/([^\s)#`"'>]+)/g;
		for (const page of pages) {
			for (const match of page.text.matchAll(blob)) {
				const path = match[1].replace(/[.,;:]+$/, "");
				expect(existsSync(join(REPO_ROOT, path)), `docs/${page.file}: ${match[0]} does not exist in the repo`).toBe(
					true,
				);
			}
		}
	});
});

describe("docs site: commands and tools match the code", () => {
	const source = (path: string) => readFileSync(join(REPO_ROOT, path), "utf8");
	const listOf = (text: string, name: string): string[] => {
		const block = new RegExp(`${name}\\s*=\\s*(?:new Set\\()?\\[([\\s\\S]*?)\\]`).exec(text);
		if (!block) throw new Error(`cannot parse ${name}`);
		return [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
	};
	const cliIndex = source("src/cli/index.ts");
	const commands = new Set(listOf(cliIndex, "COMMANDS"));
	const flags = new Set([...listOf(cliIndex, "BOOLEAN_FLAGS"), ...listOf(cliIndex, "VALUE_FLAGS")]);
	const liteSubcommands = new Set(
		[...source("src/cli/commands/lite.ts").matchAll(/case "([a-z]+)":/g)].map((m) => m[1]),
	);

	/** Shell lines and inline code that invoke the `autorag` binary. */
	function invocations(markdown: string): string[] {
		const lines: string[] = [];
		for (const block of fencedBlocks(markdown)) lines.push(...block.split("\n"));
		lines.push(...inlineCode(markdown));
		return lines;
	}

	it("only uses real `autorag` commands, lite subcommands, and flags", () => {
		let seen = 0;
		for (const page of pages) {
			for (const line of invocations(page.text)) {
				// Shell comments and sample program output are prose, not invocations.
				if (/^\s*#/.test(line)) continue;
				const match = /(?:^|[\s$(|;&])autorag\s+([a-z][a-z0-9-]*)(?![\w:])((?:\s+[^`\n]*)?)/.exec(line);
				if (!match || (match[1] === "init" && /\bnpm\b/.test(line))) continue;
				seen++;
				expect(
					commands,
					`docs/${page.file}: unknown command "autorag ${match[1]}" in \`${line.trim()}\``,
				).toContain(match[1]);
				if (match[1] === "lite") {
					const sub = /^\s+([a-z][a-z0-9-]*)/.exec(match[2])?.[1];
					if (sub) expect(liteSubcommands, `docs/${page.file}: unknown "autorag lite ${sub}"`).toContain(sub);
				}
				for (const flag of match[2].matchAll(/\s--([a-z][a-z0-9-]*)/g)) {
					expect(flags, `docs/${page.file}: unknown flag --${flag[1]} in \`${line.trim()}\``).toContain(flag[1]);
				}
			}
		}
		expect(seen, "the docs must show real autorag invocations").toBeGreaterThan(40);
	});

	it("only names MCP tools the server registers", async () => {
		const registered = new Set(await registeredMcpToolNames({ readOnly: false }));
		const FILE_SUFFIX = /\.(json|md|ts|mts|js|mjs|yaml|yml|txt|org|com|dev|app|io|test|local)$/;
		let seen = 0;
		for (const page of pages) {
			const code = [...fencedBlocks(page.text), ...inlineCode(page.text)].join("\n");
			for (const match of code.matchAll(/(?<![\w./@-])autorag\.[a-z_]+(?:\.[a-z_]+)?(?![\w@-])/g)) {
				const name = match[0];
				if (FILE_SUFFIX.test(name) && !registered.has(name)) continue;
				// Dynamic per-datasource tools are named after the configured datasource, so only the prefix is fixed.
				if (name.startsWith("autorag.search_datasource_")) continue;
				seen++;
				expect(registered, `docs/${page.file}: unknown MCP tool ${name}`).toContain(name);
			}
		}
		expect(seen, "the docs must name the MCP tools").toBeGreaterThan(5);
	});

	it("documents every builtin datasource on its own page", () => {
		for (const name of builtinDatasourceSkillNames("darwin")) {
			expect(byFile.has(`datasources/${name}.md`), `no docs/datasources/${name}.md`).toBe(true);
		}
		const overview = byFile.get("datasources/overview.md")?.text ?? "";
		for (const name of builtinDatasourceSkillNames("darwin")) {
			expect(overview, `datasources/overview.md must mention \`${name}\``).toContain(`\`${name}\``);
		}
	});

	it("documents every shipped skill", () => {
		const skills = readdirSync(join(REPO_ROOT, "skills"), { withFileTypes: true })
			.filter((entry) => entry.isDirectory())
			.map((entry) => entry.name);
		const text = byFile.get("agents/skills.md")?.text ?? "";
		for (const skill of skills) expect(text, `agents/skills.md must document the ${skill} skill`).toContain(skill);
	});
});

describe("docs site: no deprecated or wrong guidance", () => {
	const banned: ReadonlyArray<readonly [string, RegExp]> = [
		[
			"the removed gateway/model-cache commands",
			/\bautorag (gateway|models (prefetch|import|verify)|ui|serve|p2p)\b/,
		],
		[
			"pre-2026 LLM ids",
			/\b(gpt-[345][\w.-]*|o[134]-(mini|preview|pro)|claude-(3[\w.-]*|(sonnet|opus|haiku)-4[\w.-]*)|gemini-[12]\.[\w.-]*|llama[- ]?[34][\w.-]*|deepseek-(v3|r1)|qwen-?2\.5|glm-4[\w.-]*)\b/i,
		],
		[
			"the removed `spotlight` or `cloud-drive` templates as something to configure",
			/^\s*"(spotlight|cloud-drive)"\s*:/m,
		],
	];
	it.each(banned)("never mentions %s", (_label, pattern) => {
		for (const page of pages) expect(page.text, `docs/${page.file}`).not.toMatch(pattern);
	});

	it("never ships a secret-looking value", () => {
		for (const page of pages) {
			expect(page.text, `docs/${page.file}`).not.toMatch(
				/\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|xox[bap]-[A-Za-z0-9-]{10,})\b/,
			);
		}
	});
});

describe("docs site: agents and publishing", () => {
	it("generates llms.txt that points agents at the GitHub Markdown of every page", () => {
		const llms = renderLlmsTxt();
		for (const item of allPages()) {
			expect(llms).toContain(
				`https://raw.githubusercontent.com/Marker-Inc-Korea/AutoRAG/main/docs/${item.link.replace(/^\//, "")}.md`,
			);
		}
		expect(llms.split("\n")[0]).toBe("# AutoRAG Agent");
	});

	it("serves the custom domain and tells agents to read the Markdown", () => {
		expect(readFileSync(join(DOCS_ROOT, "public/CNAME"), "utf8").trim()).toBe(new URL(SITE).host);
		const readme = byFile.get("README.md")?.text ?? "";
		expect(readme).toContain(SITE);
		expect(readme).toContain(".vitepress/sidebar.mts");
		expect(readme).toMatch(/Markdown/);
		expect(byFile.get("index.md")?.text ?? "").toContain("layout: home");
	});

	it("is linked from the root README as the documentation home", () => {
		const readme = readFileSync(join(REPO_ROOT, "README.md"), "utf8");
		expect(readme).toContain(SITE);
		expect(readme).toMatch(
			/\]\(docs\/README\.md\)|\]\(https:\/\/github\.com\/Marker-Inc-Korea\/AutoRAG\/tree\/main\/docs\)/,
		);
	});

	it("keeps every `docs/...md` reference in the repo pointing at a page that exists", () => {
		const roots = [
			"README.md",
			"AGENTS.md",
			"CONTRIBUTING.md",
			"SECURITY.md",
			"MAINTAINERS",
			"GOVERNANCE.md",
			"plugins",
			"skills",
			"internal-skills",
			"src",
			"scripts",
			".github",
		];
		const stale: string[] = [];
		const files: string[] = [];
		const walk = (dir: string): void => {
			for (const entry of readdirSync(dir, { withFileTypes: true })) {
				if (entry.name === "node_modules" || entry.name === ".git") continue;
				const full = join(dir, entry.name);
				if (entry.isDirectory()) walk(full);
				else if (/\.(md|ts|mts|mjs|json|yml|yaml)$|^MAINTAINERS$/.test(entry.name)) files.push(full);
			}
		};
		for (const root of roots) {
			const full = join(REPO_ROOT, root);
			if (!existsSync(full)) continue;
			if (statSync(full).isDirectory()) walk(full);
			else files.push(full);
		}
		for (const file of files) {
			const text = readFileSync(file, "utf8");
			for (const match of text.matchAll(/(?<![\w/.-])docs\/((?:[a-z0-9_-]+\/)*[a-z0-9_-]+\.md)(?:#[\w-]+)?/g)) {
				// Test fixtures and sample corpora use a `docs/` folder of their own.
				if (file.includes(`${join("scripts", "manual-qa")}`) && match[1] === "README.md") continue;
				if (!existsSync(join(DOCS_ROOT, match[1]))) stale.push(`${file.slice(REPO_ROOT.length)}: docs/${match[1]}`);
			}
		}
		expect(stale).toEqual([]);
	});
});
