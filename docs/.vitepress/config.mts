import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig } from "vitepress";
import { REPO, renderLlmsTxt, SITE } from "./llms.mts";
import { SECTIONS } from "./sidebar.mts";

const sidebar = Object.fromEntries(
	SECTIONS.map((section) => [section.base, section.groups.map((group) => ({ ...group, items: [...group.items] }))]),
);

export default defineConfig({
	title: "AutoRAG",
	titleTemplate: ":title | AutoRAG",
	description:
		"Developer documentation for AutoRAG Agent: a personal librarian that finds, reads, and cites your files, chats, and mail.",
	lang: "en-US",
	cleanUrls: true,
	lastUpdated: true,
	// README.md is the GitHub entry point for this folder, not a site page.
	srcExclude: ["README.md", "node_modules/**"],
	head: [
		["link", { rel: "icon", href: "/logo.svg", type: "image/svg+xml" }],
		["meta", { name: "theme-color", content: "#38bdf8" }],
		["meta", { property: "og:type", content: "website" }],
		["meta", { property: "og:site_name", content: "AutoRAG" }],
		["meta", { property: "og:image", content: `${SITE}/img/hero.webp` }],
	],
	sitemap: { hostname: SITE },
	themeConfig: {
		logo: "/logo.svg",
		siteTitle: "AutoRAG",
		nav: [
			...SECTIONS.map((section) => ({
				text: section.text,
				link: section.groups[0].items[0].link,
				activeMatch: section.base,
			})),
			{ text: "Legacy (AutoML)", link: `${SITE}/legacy/` },
		],
		sidebar,
		search: { provider: "local" },
		socialLinks: [{ icon: "github", link: REPO }],
		editLink: { pattern: `${REPO}/edit/main/docs/:path`, text: "Edit this page on GitHub" },
		outline: { level: [2, 3], label: "On this page" },
		footer: {
			message: "Released under the Apache-2.0 license.",
			copyright: "Copyright © Marker Inc.",
		},
	},
	markdown: { lineNumbers: false },
	// Dead links fail the build (VitePress default): the docs are code, and a broken link is a bug.
	async buildEnd(siteConfig) {
		mkdirSync(siteConfig.outDir, { recursive: true });
		writeFileSync(join(siteConfig.outDir, "llms.txt"), renderLlmsTxt());
	},
});
