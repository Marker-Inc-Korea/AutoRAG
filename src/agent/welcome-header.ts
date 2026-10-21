import type { ExtensionFactory, SettingsManager, Theme } from "@earendil-works/pi-coding-agent";
import { type Component, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

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

/**
 * One is shown per session, picked at random. Each fits on one line of an
 * 80-column terminal after the `Tip · ` label, and is true of the TUI as shipped.
 */
export const WELCOME_TIPS: readonly string[] = [
	"Ask in plain language; answers cite the files they came from.",
	"Press Enter during a search to steer it, Alt+Enter to queue a follow-up.",
	"Esc interrupts a running search.",
	"/embedding on starts the local embedding daemon (Ctrl+Shift+E toggles).",
	"/model switches the model, /login signs in to a provider.",
	"/new starts a fresh session, /resume returns to an earlier one.",
	"/hotkeys lists every keyboard shortcut.",
];

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
 * The first screen of `autorag tui`: the logo, the version, and one random tip,
 * centered on `width`. The logo is centered as one block so its shape holds;
 * the version and the tip are centered on their own. The tip is always one line
 * and is cut with `...` when the terminal is narrower than it.
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
	const tip = WELCOME_TIPS[Math.floor(random() * WELCOME_TIPS.length)] as string;
	lines.push(center(`${theme.fg("accent", "Tip")} ${theme.fg("dim", "·")} ${theme.fg("muted", tip)}`, width));
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
