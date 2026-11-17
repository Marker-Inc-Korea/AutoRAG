import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { defineConfig } from "vitepress";
import { REPO, renderLlmsTxt, SITE } from "./llms.mts";
import { SECTIONS } from "./sidebar.mts";

/**
 * One sidebar for the whole site, so every section is reachable from every page at every width (VitePress only shows
 * the top nav above 768px, and a per-section sidebar would hide the other sections on tablets and phones).
 */
const sidebar = SECTIONS.map((section) => ({
	text: section.text,
	collapsed: section.base !== "/guide/",
	items:
		section.groups.length === 1
			? section.groups[0].items.map((item) => ({ ...item }))
			: section.groups.map((group) => ({ text: group.text, items: group.items.map((item) => ({ ...item })) })),
})).concat({
	text: "Legacy (Python AutoML)",
	collapsed: false,
	items: [{ text: "Original AutoRAG docs", link: `${SITE}/legacy/` }],
});

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
		["link", { rel: "icon", href: "/favicon.png", type: "image/png" }],
		["link", { rel: "apple-touch-icon", href: "/apple-touch-icon.png" }],
		["meta", { name: "theme-color", content: "#38bdf8" }],
		["meta", { property: "og:type", content: "website" }],
		["meta", { property: "og:site_name", content: "AutoRAG" }],
		["meta", { property: "og:image", content: `${SITE}/img/social-preview.png` }],
	],
	sitemap: { hostname: SITE },
	themeConfig: {
		logo: "/logo.png",
		siteTitle: "AutoRAG",
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
