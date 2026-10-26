import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type ExtensionContext, SettingsManager } from "@earendil-works/pi-coding-agent";
import { stripTerminalSequences, visibleWidth } from "@earendil-works/pi-tui";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createWelcomeHeader, renderWelcomeBanner, WELCOME_TIPS } from "../../src/agent/welcome-header.ts";

/** A theme that leaves text untouched, so rendered lines can be measured as typed. */
const plain = { fg: (_color: string, text: string) => text, bold: (text: string) => text };

const banner = (width: number, random: () => number = () => 0) =>
	renderWelcomeBanner({ width, version: "9.9.9", theme: plain as never, random }).map((line) =>
		stripTerminalSequences(line),
	);

/** Leading spaces of a line. */
const indent = (line: string) => line.length - line.trimStart().length;

describe("renderWelcomeBanner", () => {
	it("never exceeds the terminal width, however narrow", () => {
		for (const width of [200, 100, 80, 60, 59, 40, 37, 36, 20, 10, 5, 1]) {
			for (const line of banner(width))
				expect(visibleWidth(line), `width ${width}: "${line}"`).toBeLessThanOrEqual(width);
		}
	});

	it("draws the big logo when it fits and a smaller one, then a plain wordmark, as the terminal narrows", () => {
		const big = banner(100).filter((line) => line.includes("█"));
		expect(big.length).toBeGreaterThanOrEqual(5);
		expect(banner(50).some((line) => line.includes("█"))).toBe(false);
		expect(banner(50).some((line) => line.includes("/_/ |_"))).toBe(true);
		const tiny = banner(20);
		expect(tiny.some((line) => line.includes("AutoRAG"))).toBe(true);
		expect(tiny.some((line) => /[█╔╗╚╝]|\/_\//.test(line))).toBe(false);
	});

	it("centers the logo as one block so its shape is kept", () => {
		const width = 101;
		const logo = banner(width).filter((line) => line.includes("█"));
		const indents = new Set(logo.map(indent));
		// Rows start at different columns inside the art, but the block's left edge is one column:
		// the top row starts one cell right of the others, so the smallest indent is the block's.
		const left = Math.min(...logo.map(indent));
		const blockWidth = Math.max(...logo.map((line) => line.trimEnd().length - left));
		expect(left).toBe(Math.floor((width - blockWidth) / 2));
		expect(indents.size).toBeGreaterThan(0);
	});

	it("centers the version line and the tip on their own", () => {
		const width = 120;
		const lines = banner(width).filter((line) => line.trim().length > 0);
		const centered = lines.filter((line) => line.includes("v9.9.9") || line.includes("Tip"));
		expect(centered).toHaveLength(2);
		for (const line of centered) {
			const left = indent(line);
			const right = width - left - line.trim().length;
			expect(Math.abs(left - right), `"${line}"`).toBeLessThanOrEqual(1);
		}
	});

	it("shows the version and tips, and leaves pi's skills and key-hint listing out", () => {
		const text = banner(120).join("\n");
		expect(text).toContain("v9.9.9");
		expect(text).toMatch(/Tip/);
		expect(text).not.toMatch(/\[Skills\]|to interrupt|to cycle models|Pi can explain/i);
	});

	it("shows exactly one tip, picked by the random source", () => {
		const tipLines = (random: () => number) => banner(160, random).filter((line) => line.includes("Tip"));
		expect(tipLines(() => 0)).toHaveLength(1);
		expect(tipLines(() => 0.999)).toHaveLength(1);
		expect(tipLines(() => 0)).not.toEqual(tipLines(() => 0.999));
		const seen = WELCOME_TIPS.map((_, index) => tipLines(() => index / WELCOME_TIPS.length)[0]?.trim());
		expect(new Set(seen).size).toBe(WELCOME_TIPS.length);
	});

	it("keeps every tip on one line on an 80-column terminal, uncut", () => {
		for (const [index, tip] of WELCOME_TIPS.entries()) {
			const lines = banner(80, () => index / WELCOME_TIPS.length);
			const tipLines = lines.filter((line) => line.includes("Tip"));
			expect(tipLines, tip).toHaveLength(1);
			expect(tipLines[0]?.trim(), tip).toBe(`Tip · ${tip}`);
		}
	});

	it("cuts a tip wider than the terminal to one line instead of wrapping it", () => {
		const lines = banner(30, () => 0);
		for (const line of lines) expect(visibleWidth(line)).toBeLessThanOrEqual(30);
		const tipLines = lines.filter((line) => line.includes("Tip"));
		expect(tipLines).toHaveLength(1);
		expect(tipLines[0]).toMatch(/\.\.\.$/);
	});
});

type HeaderFactory = (tui: unknown, theme: unknown) => { render(width: number): string[] };

/** A pi that records the `session_start` handler, and a ctx whose ui records `setHeader`. */
function install(settings: SettingsManager, options: { hasUI?: boolean } = {}) {
	const handlers = new Map<string, (event: unknown, ctx: ExtensionContext) => void>();
	const headers: HeaderFactory[] = [];
	const ctx = {
		hasUI: options.hasUI ?? true,
		ui: { setHeader: (factory: HeaderFactory) => headers.push(factory) },
	} as unknown as ExtensionContext;
	// Each call yields a new offset, so a banner that re-rolled its tips per render would show it.
	let calls = 0;
	createWelcomeHeader({ version: "9.9.9", settings, random: () => (calls++ % 7) / 7 })({
		on: (event: string, handler: (event: unknown, ctx: ExtensionContext) => void) =>
			void handlers.set(event, handler),
	} as never);
	return {
		headers,
		start: (reason: string) => handlers.get("session_start")?.({ type: "session_start", reason }, ctx),
	};
}

let root: string;
let agentDir: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-welcome-"));
	agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function settingsWith(global?: Record<string, unknown>, project?: Record<string, unknown>): SettingsManager {
	if (global !== undefined) writeFileSync(join(agentDir, "settings.json"), JSON.stringify(global));
	if (project !== undefined) {
		mkdirSync(join(root, ".pi"), { recursive: true });
		writeFileSync(join(root, ".pi", "settings.json"), JSON.stringify(project));
	}
	return SettingsManager.create(root, agentDir);
}

describe("createWelcomeHeader", () => {
	// pi restores its own header on every session switch, so the banner must follow every start.
	it.each(["startup", "new", "reload", "resume", "fork"])(
		"installs the banner when a session starts (%s)",
		(reason) => {
			const pi = install(settingsWith());
			pi.start(reason);
			expect(pi.headers).toHaveLength(1);
			const component = (pi.headers[0] as HeaderFactory)({}, plain);
			expect(component.render(120).map(stripTerminalSequences).join("\n")).toContain("v9.9.9");
		},
	);

	it("does nothing without a UI (print and RPC modes)", () => {
		const settings = settingsWith();
		const pi = install(settings, { hasUI: false });
		pi.start("startup");
		expect(pi.headers).toHaveLength(0);
		expect(settings.getQuietStartup()).toBe(false);
	});

	it("draws against the width pi gives it at render time, so a resize re-centers", () => {
		const pi = install(settingsWith());
		pi.start("startup");
		const component = (pi.headers[0] as HeaderFactory)({}, plain);
		const logoLeft = (width: number) =>
			Math.min(
				...component
					.render(width)
					.map(stripTerminalSequences)
					.filter((line) => line.includes("█"))
					.map(indent),
			);
		expect(logoLeft(140)).toBeGreaterThan(logoLeft(100));
	});

	it("keeps the same tip across a resize", () => {
		const pi = install(settingsWith());
		pi.start("startup");
		const component = (pi.headers[0] as HeaderFactory)({}, plain);
		const tip = (width: number) =>
			component
				.render(width)
				.map((line) => stripTerminalSequences(line).trim())
				.filter((line) => line.startsWith("Tip"));
		const first = tip(200);
		expect(first).toHaveLength(1);
		expect(tip(160)).toEqual(first);
		expect(tip(200)).toEqual(first);
	});

	it("hides pi's loaded-resource listing (skills, prompts, extensions) by default, keeping the header", () => {
		const settings = settingsWith();
		install(settings).start("startup");
		expect(settings.getQuietStartup()).toBe("header");
	});

	it("keeps a quietStartup the user saved: true shows no banner at all", () => {
		const settings = settingsWith({ quietStartup: true });
		const pi = install(settings);
		pi.start("startup");
		expect(settings.getQuietStartup()).toBe(true);
		expect(pi.headers).toHaveLength(0);
	});

	it("keeps a saved quietStartup: false, so the user's own choice to list resources stands", () => {
		const settings = settingsWith({ quietStartup: false });
		const pi = install(settings);
		pi.start("startup");
		expect(settings.getQuietStartup()).toBe(false);
		expect(pi.headers).toHaveLength(1);
	});

	it("lets a project quietStartup win over the AutoRAG default", () => {
		const settings = settingsWith(undefined, { quietStartup: true });
		const pi = install(settings);
		pi.start("startup");
		expect(pi.headers).toHaveLength(0);
	});

	it("re-applies the default after /reload drops pi's overrides", async () => {
		const settings = settingsWith();
		const pi = install(settings);
		pi.start("startup");
		await settings.reload();
		expect(settings.getQuietStartup()).toBe(false);
		pi.start("reload");
		expect(settings.getQuietStartup()).toBe("header");
	});
});
