import type { ExtensionAPI, ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { matchesKey } from "@earendil-works/pi-tui";
import type { QueryRoute } from "./query-routing.ts";

/**
 * Interactive / CLI / MCP route selection mode.
 *
 * - `auto` (default): Jev classifies each Turn-1 query into `local`, `organize`,
 *   `extra`, or `config`.
 * - `local`: explicit search / summarize mode — skips Jev's branch classification
 *   and runs the local + datasource retrieval pipeline directly.
 * - `organize`: explicit file-organization mode — skips Jev's branch classification
 *   and runs a direct file-organization turn with the packaged organize skill.
 * - `extra`: explicit direct-task / general-answer mode — skips Jev's branch
 *   classification and runs one tool-capable turn without retrieval.
 * - `config`: explicit AutoRAG configuration / automation mode — skips Jev's
 *   branch classification and runs a self-config turn with the setup skill.
 */
export type RouteMode = "auto" | QueryRoute;

/** Canonical order Ctrl+P cycles through when self-config is enabled. */
export const ROUTE_MODES: readonly RouteMode[] = ["auto", "local", "organize", "extra", "config"];

/** Route modes available when self-config (`config` branch) is disabled. */
export const BASE_ROUTE_MODES: readonly RouteMode[] = ["auto", "local", "organize", "extra"];

/** Default route mode for CLI, TUI, and MCP: Jev decides. */
export const DEFAULT_ROUTE_MODE: RouteMode = "auto";

/** Key that cycles the active route mode in the TUI. */
export const ROUTE_MODE_CYCLE_KEY = "ctrl+p";

/** Footer slot for the active route mode indicator. */
export const ROUTE_MODE_STATUS_KEY = "autorag-route-mode";

const ROUTE_MODE_ALIASES: Readonly<Record<string, RouteMode>> = {
	auto: "auto",
	jev: "auto",
	local: "local",
	find: "local",
	search: "local",
	summarize: "local",
	explore: "local",
	organize: "organize",
	extra: "extra",
	direct: "extra",
	config: "config",
	automate: "config",
};

const MODE_SUMMARY: Readonly<Record<RouteMode, string>> = {
	auto: "Jev decides the route automatically (default)",
	local: "search & summarize local files and datasources (skip Jev routing)",
	organize: "organize, rename, move, or convert files directly (skip Jev routing)",
	extra: "direct answer or general tool task without retrieval (skip Jev routing)",
	config: "configure AutoRAG settings and automations (skip Jev routing)",
};

const MODE_LABEL: Readonly<Record<RouteMode, string>> = {
	auto: "mode: auto",
	local: "mode: local",
	organize: "mode: organize",
	extra: "mode: extra",
	config: "mode: config",
};

/**
 * Parse a route mode string case-insensitively.
 *
 * Accepts canonical values (`auto`, `local`, `organize`, `extra`, `config`) and
 * Dazzi-app / user-friendly aliases (`find`, `search`, `summarize`, `explore`,
 * `direct`, `automate`, `jev`). Returns `undefined` for unrecognized values.
 */
export function parseRouteMode(value: unknown): RouteMode | undefined {
	if (typeof value !== "string") return undefined;
	const normalized = value.trim().toLowerCase();
	if (normalized === "") return undefined;
	return ROUTE_MODE_ALIASES[normalized];
}

/**
 * Convert a {@link RouteMode} into the Turn-1 {@link QueryRoute} override passed
 * to `AutoRAGAgent.searchDocuments` / `searchDocumentsStream`. `auto` maps to
 * `undefined` so Jev performs branch routing as usual.
 */
export function routeOverrideForMode(mode: RouteMode): QueryRoute | undefined {
	return mode === "auto" ? undefined : mode;
}

export function availableRouteModes(selfConfigEnabled = true): readonly RouteMode[] {
	return selfConfigEnabled ? ROUTE_MODES : BASE_ROUTE_MODES;
}

export function cycleRouteMode(mode: RouteMode, selfConfigEnabled = true): RouteMode {
	const modes = availableRouteModes(selfConfigEnabled);
	const index = modes.indexOf(mode);
	return modes[(index + 1) % modes.length] as RouteMode;
}

export function describeRouteMode(mode: RouteMode): string {
	return MODE_SUMMARY[mode];
}

export function routeModeStatusLabel(mode: RouteMode): string {
	return MODE_LABEL[mode];
}

type RouteModeListener = (mode: RouteMode) => void;

/**
 * Session-wide route mode state (`auto` or an explicit {@link QueryRoute}).
 * Shared across the TUI controls (`Ctrl+P` / `/mode`) and `AutoRAGAgent` so
 * switching mode takes effect immediately for the next submitted turn.
 */
export class RouteModeController {
	private current: RouteMode;
	private readonly selfConfigEnabled: boolean;
	private readonly listeners = new Set<RouteModeListener>();

	constructor(mode: RouteMode = DEFAULT_ROUTE_MODE, options: { readonly selfConfigEnabled?: boolean } = {}) {
		this.selfConfigEnabled = options.selfConfigEnabled !== false;
		this.current = mode === "config" && !this.selfConfigEnabled ? DEFAULT_ROUTE_MODE : mode;
	}

	get mode(): RouteMode {
		return this.current;
	}

	get routeOverride(): QueryRoute | undefined {
		return routeOverrideForMode(this.current);
	}

	get availableModes(): readonly RouteMode[] {
		return availableRouteModes(this.selfConfigEnabled);
	}

	set(mode: RouteMode): void {
		const effective = mode === "config" && !this.selfConfigEnabled ? DEFAULT_ROUTE_MODE : mode;
		if (effective === this.current) return;
		this.current = effective;
		for (const listener of [...this.listeners]) listener(effective);
	}

	cycle(): RouteMode {
		this.set(cycleRouteMode(this.current, this.selfConfigEnabled));
		return this.current;
	}

	subscribe(listener: RouteModeListener): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}
}

export interface RouteModeControlsOptions {
	readonly controller: RouteModeController;
}

/**
 * pi extension that displays the active route mode in the TUI footer and lets
 * the user switch between `auto` (Jev routing) and explicit routes (`local`,
 * `organize`, `extra`, `config`) via `Ctrl+P` or `/mode [auto|local|organize|extra|config]`.
 *
 * Uses `ui.onTerminalInput` to intercept `Ctrl+P` before pi's editor-global
 * `app.model.cycleForward` binding (which is reserved against `registerShortcut`).
 */
export function createRouteModeControls(options: RouteModeControlsOptions): ExtensionFactory {
	const { controller } = options;

	return (pi: ExtensionAPI) => {
		let boundUi: ExtensionContext["ui"] | undefined;
		let unsubscribeKeys: (() => void) | undefined;
		let unsubscribeMode: (() => void) | undefined;

		const showStatus = (ui: ExtensionContext["ui"], mode: RouteMode): void => {
			const text = routeModeStatusLabel(mode);
			const color = mode === "auto" ? "muted" : "accent";
			ui.setStatus(ROUTE_MODE_STATUS_KEY, ui.theme?.fg ? ui.theme.fg(color, text) : text);
		};

		const bindUi = (ui: ExtensionContext["ui"]): void => {
			showStatus(ui, controller.mode);
			if (boundUi === ui) return;
			unsubscribeKeys?.();
			unsubscribeMode?.();
			boundUi = ui;
			unsubscribeMode = controller.subscribe((mode) => showStatus(ui, mode));
			unsubscribeKeys = ui.onTerminalInput((data) => {
				if (!matchesKey(data, ROUTE_MODE_CYCLE_KEY)) return undefined;
				const mode = controller.cycle();
				ui.notify(`Route mode: ${mode} — ${MODE_SUMMARY[mode]}`, "info");
				return { consume: true };
			});
		};

		pi.on("session_start", (_event, ctx) => {
			if (ctx.hasUI) bindUi(ctx.ui);
		});

		pi.on("session_shutdown", () => {
			unsubscribeKeys?.();
			unsubscribeMode?.();
			unsubscribeKeys = undefined;
			unsubscribeMode = undefined;
			boundUi = undefined;
		});

		const modes = controller.availableModes;

		pi.registerCommand("mode", {
			description: `Show or set route mode: ${modes.join(", ")} (Ctrl+P cycles; auto uses Jev routing)`,
			getArgumentCompletions: (prefix) => {
				const normalized = prefix.trim().toLowerCase();
				const items = modes
					.filter((mode) => mode.startsWith(normalized))
					.map((mode) => ({
						value: mode,
						label: mode,
						description: MODE_SUMMARY[mode],
					}));
				return items.length > 0 ? items : null;
			},
			handler: async (args, ctx) => {
				const wanted = args.trim();
				let mode = parseRouteMode(wanted);
				if (wanted !== "" && (mode === undefined || !modes.includes(mode))) {
					ctx.ui.notify(`Unknown route mode "${wanted}". Use one of: ${modes.join(", ")}.`, "warning");
					return;
				}
				if (mode === undefined) {
					const choice = await ctx.ui.select(
						`Route mode (now ${controller.mode})`,
						modes.map((candidate) => `${candidate} - ${MODE_SUMMARY[candidate]}`),
					);
					mode = parseRouteMode(choice?.split(" ")[0]);
					if (mode === undefined) return;
				}
				controller.set(mode);
				ctx.ui.notify(`Route mode: ${mode} — ${MODE_SUMMARY[mode]}`, "info");
			},
		});
	};
}
