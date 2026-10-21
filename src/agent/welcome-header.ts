import type { ExtensionFactory, SettingsManager, Theme } from "@earendil-works/pi-coding-agent";
import { type Component, truncateToWidth, visibleWidth, wrapTextWithAnsi } from "@earendil-works/pi-tui";

/** Logos from widest to narrowest. The first whose block fits the terminal is drawn. */
const LOGOS: readonly (readonly string[])[] = [
	[
		" █████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  █████╗  ██████╗ ",
		"██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔══██╗██╔══██╗██╔════╝ ",
		"███████║██║   ██║   ██║   ██║   ██║██████╔╝███████║██║  ███╗",
		"██╔══██║██║   ██║   ██║   ██║   ██║██╔══██╗██╔══██║██║   ██║",
		"██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║  ██║██║  ██║╚██████╔╝",
		"╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ",
	],
	[
		"   ___       __       ___  ___  _____",
		"  / _ |__ __/ /____  / _ \\/ _ |/ ___/",
		" / __ / // / __/ _ \\/ , _/ __ / (_ / ",
		"/_/ |_\\_,_/\\__/\\___/_/|_/_/ |_\\___/  ",
	],
].map((rows) => rows.map((row) => row.trimEnd()));

const WORDMARK = "AutoRAG";

/** Shown three at a time, rotating. Each one is true of the TUI as shipped; none contains the word the banner prefixes them with. */
export const WELCOME_TIPS: readonly string[] = [
	"Ask in plain language: AutoRAG searches your files and sources and cites what it finds.",
	"Press Enter while a search runs to steer it; Alt+Enter queues a follow-up question.",
	"Esc interrupts a running search.",
	"/embedding on starts the local embedding daemon; Ctrl+Shift+E toggles it.",
	"/model switches the model, /login signs in to a provider.",
	"/new starts a fresh session, /resume returns to an earlier one.",
	"/hotkeys lists every keyboard shortcut.",
];

const TIPS_SHOWN = 3;

export interface WelcomeBannerOptions {
	readonly width: number;
	readonly version: string;
	readonly theme: Pick<Theme, "fg" | "bold">;
	/** Source of the rotating tip offset. Default `Math.random`. */
	readonly random?: () => number;
}

/** Centers one already-styled line; a line wider than the terminal is cut. */
function center(styled: string, width: number): string {
	const fitted = truncateToWidth(styled, width);
	return " ".repeat(Math.max(0, Math.floor((width - visibleWidth(fitted)) / 2))) + fitted;
}

/**
 * The first screen of `autorag tui`: the logo, the version, and a few tips,
 * centered on `width`. The logo is centered as one block so its shape holds;
 * the version and each tip line are centered on their own.
 */
export function renderWelcomeBanner({ width, version, theme, random = Math.random }: WelcomeBannerOptions): string[] {
	const logo = LOGOS.find((rows) => Math.max(...rows.map((row) => row.length)) <= width);
	const lines: string[] = [""];
	if (logo === undefined) {
		lines.push(center(theme.bold(theme.fg("accent", WORDMARK)), width));
	} else {
		const left = " ".repeat(Math.floor((width - Math.max(...logo.map((row) => row.length))) / 2));
		for (const row of logo) lines.push(left + theme.fg("accent", row));
	}
	lines.push("", center(theme.fg("muted", `v${version} · personal librarian agent`), width), "");
	const start = Math.floor(random() * WELCOME_TIPS.length);
	for (let offset = 0; offset < TIPS_SHOWN; offset++) {
		const tip = WELCOME_TIPS[(start + offset) % WELCOME_TIPS.length] as string;
		const label = `${theme.fg("accent", "Tip")} ${theme.fg("dim", "·")} `;
		// Wrap the tip itself to what the label leaves, so continuation lines stay inside the terminal.
		const body = wrapTextWithAnsi(theme.fg("muted", tip), Math.max(1, width - visibleWidth(label)));
		for (const [index, row] of body.entries()) lines.push(center(index === 0 ? label + row : row, width));
	}
	lines.push("");
	return lines;
}

/** The slice of pi's settings the banner reads and adjusts. */
type WelcomeSettings = Pick<
	SettingsManager,
	"applyOverrides" | "getGlobalSettings" | "getProjectSettings" | "getQuietStartup"
>;

export interface WelcomeHeaderOptions {
	readonly version: string;
	readonly settings: WelcomeSettings;
	readonly random?: () => number;
}

/**
 * Gives AutoRAG's startup the `"header"` quiet level (header on, pi's loaded
 * skills/prompts/extensions listing off) unless the user saved their own
 * `quietStartup` in pi's global or project settings. pi rebuilds its settings
 * from disk on `/reload`, dropping overrides, so this runs on every session start.
 */
function applyQuietDefault(settings: WelcomeSettings): void {
	if (settings.getProjectSettings().quietStartup !== undefined) return;
	if (settings.getGlobalSettings().quietStartup !== undefined) return;
	settings.applyOverrides({ quietStartup: "header" });
}

/**
 * Replaces pi's startup header (its logo, key hints) with the AutoRAG banner on
 * every session start: launch, `/new`, `/resume`, `/fork`, `/reload`. pi
 * restores its built-in header before each session start, so there is nothing
 * to clean up. A user who saved `quietStartup: true` gets no banner.
 */
export function createWelcomeHeader(options: WelcomeHeaderOptions): ExtensionFactory {
	return (pi) => {
		pi.on("session_start", (_event, ctx) => {
			if (!ctx.hasUI) return;
			applyQuietDefault(options.settings);
			if (options.settings.getQuietStartup() === true) return;
			// Fixed per session so a resize re-centers the same tips instead of reshuffling them.
			const offset = (options.random ?? Math.random)();
			ctx.ui.setHeader(
				(_tui, theme): Component => ({
					render: (width) => renderWelcomeBanner({ width, version: options.version, theme, random: () => offset }),
					invalidate: () => undefined,
				}),
			);
		});
	};
}
