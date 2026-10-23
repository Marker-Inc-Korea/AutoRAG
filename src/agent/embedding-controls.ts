import type { ExtensionAPI, ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { ensureMinSyncDaemon, probeMinSyncDaemon, stopMinSyncDaemon } from "../embedding/daemon.ts";
import type { EmbedderHealth, ManagedEmbedder } from "../embedding/embedder.ts";

/** The footer and notifications of one pi session; writes after that session is replaced are dropped. */
interface SessionFooter {
	setStatus(text: string | undefined): void;
	notify(message: string, type: "info" | "warning" | "error"): void;
}

/** Footer slot of the embedding indicator. */
export const EMBEDDING_STATUS_KEY = "autorag-embedding";
/** Toggles the local `minsync serve` daemon. Not bound by pi, pi-tui or AutoRAG's own interrupt/follow-up keys. */
export const EMBEDDING_TOGGLE_KEY = "ctrl+shift+e";
const POLL_INTERVAL_MS = 15_000;

/** What `/embedding` accepts. `toggle` is also what the key does. */
const EMBEDDING_ACTIONS = ["on", "off", "toggle", "status"] as const;
type EmbeddingAction = (typeof EMBEDDING_ACTIONS)[number];

/** A sentence for `/embedding status`; the footer carries the short form. */
function describeHealth(health: EmbedderHealth): string {
	if (health.alive)
		return `Embedding backend: ${health.backend}${health.detail === undefined ? "" : ` (${health.detail})`}.`;
	return `No embedding backend is alive${health.detail === undefined ? "" : `: ${health.detail}`}. Run /embedding on, or set minSync.embedder to a remote endpoint.`;
}

export interface EmbeddingControlsOptions {
	readonly embedder: ManagedEmbedder;
	/** True when `minSync.embedder` names a remote endpoint: the toggle then has nothing local to start or stop. */
	readonly remote: boolean;
	/** Locates the `minsync` executable the toggle starts the daemon with. */
	readonly resolveBinary: () => Promise<string>;
}

/** The footer text for one probe result. Plain text: pi's footer owns colour. */
export function embeddingStatusText(health: EmbedderHealth, toggleKey: string = EMBEDDING_TOGGLE_KEY): string {
	if (health.backend === "remote") {
		return health.alive ? "embedding: remote ● up" : `embedding: remote ○ down (${health.detail ?? "no answer"})`;
	}
	if (health.backend === "daemon") return "embedding: daemon ● up";
	if (health.backend === "in-process") return "embedding: in-process ●";
	return `embedding: off ○ (/embedding on, or ${toggleKey}, starts minsync serve)`;
}

/**
 * pi extension that shows whether embedding is alive in the footer and, for the local daemon, toggles it with
 * a key. Everything is user-driven: no daemon is started or stopped except by the key, so there is no
 * lifecycle to design. The footer re-probes every {@link POLL_INTERVAL_MS} (the daemon's own idle timeout or
 * a launchd restart changes it behind our back) and right after every toggle.
 */
export function createEmbeddingControls(options: EmbeddingControlsOptions): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		let timer: NodeJS.Timeout | undefined;
		let busy = false;
		// pi invalidates an extension ctx once its session is replaced (/new, /resume, reload) and throws when
		// it is used again. A probe or a toggle can finish after that, so every write is tied to the session
		// that started it and dropped when that session is gone; the next session_start installs a fresh one.
		let generation = 0;

		const ui = (ctx: ExtensionContext): SessionFooter => {
			const mine = generation;
			return {
				setStatus: (text: string | undefined): void => {
					if (generation === mine) ctx.ui.setStatus(EMBEDDING_STATUS_KEY, text);
				},
				notify: (message: string, type: "info" | "warning" | "error"): void => {
					if (generation === mine) ctx.ui.notify(message, type);
				},
			};
		};

		const refresh = async (target: SessionFooter): Promise<void> => {
			try {
				target.setStatus(embeddingStatusText(await options.embedder.health()));
			} catch {
				target.setStatus("embedding: unknown");
			}
		};

		pi.on("session_start", (_event, ctx) => {
			if (!ctx.hasUI) return;
			generation += 1;
			const target = ui(ctx);
			void refresh(target);
			clearInterval(timer);
			timer = setInterval(() => {
				if (!busy) void refresh(target);
			}, POLL_INTERVAL_MS);
			timer.unref();
		});
		pi.on("session_shutdown", () => {
			generation += 1;
			clearInterval(timer);
			timer = undefined;
		});

		/** One action for the key and for `/embedding`: `toggle` flips, `on`/`off` only move toward that state. */
		const act = async (action: EmbeddingAction, ctx: ExtensionContext): Promise<void> => {
			const target = ui(ctx);
			if (action === "status") {
				try {
					const health = await options.embedder.health();
					target.setStatus(embeddingStatusText(health));
					target.notify(describeHealth(health), health.alive ? "info" : "warning");
				} catch (error) {
					target.notify(error instanceof Error ? error.message : String(error), "error");
				}
				return;
			}
			if (options.remote) {
				target.notify("Embedding uses a remote endpoint; there is no local daemon to start or stop.", "info");
				return;
			}
			if (busy) {
				target.notify("The embedding daemon is already being started or stopped.", "info");
				return;
			}
			busy = true;
			try {
				const running = await probeMinSyncDaemon();
				if (running !== undefined && action === "on") {
					target.notify(`minsync serve is already running at ${running.baseUrl}.`, "info");
				} else if (running === undefined && action === "off") {
					target.notify("minsync serve is not running.", "info");
				} else if (running !== undefined) {
					const stopped = await stopMinSyncDaemon();
					target.notify(`Stopped minsync serve (pid ${stopped?.pid ?? "?"}).`, "info");
				} else {
					target.setStatus("embedding: starting… (first start may download the model)");
					const binaryPath = await options.resolveBinary();
					// idle-timeout 0: the user turned it on, so it stays on until they turn it off.
					const started = await ensureMinSyncDaemon({ binaryPath, idleTimeoutSeconds: 0 });
					target.notify(`Started minsync serve at ${started.baseUrl}.`, "info");
				}
			} catch (error) {
				target.notify(error instanceof Error ? error.message : String(error), "error");
			} finally {
				busy = false;
				await refresh(target);
			}
		};

		pi.registerShortcut(EMBEDDING_TOGGLE_KEY, {
			description: "Start or stop the local embedding daemon (minsync serve)",
			handler: (ctx) => act("toggle", ctx),
		});

		pi.registerCommand("embedding", {
			description: "Embedding daemon: /embedding [on|off|toggle|status] (no argument: toggle)",
			getArgumentCompletions: (prefix) => {
				const matches = EMBEDDING_ACTIONS.filter((name) => name.startsWith(prefix.trim().toLowerCase()));
				return matches.length === 0 ? null : matches.map((name) => ({ value: name, label: name }));
			},
			handler: async (args, ctx) => {
				const word = args.trim().toLowerCase();
				const action = word === "" ? "toggle" : EMBEDDING_ACTIONS.find((name) => name === word);
				if (action === undefined) {
					ui(ctx).notify(
						`Unknown argument "${word}". Use /embedding [${EMBEDDING_ACTIONS.join("|")}].`,
						"warning",
					);
					return;
				}
				await act(action, ctx);
			},
		});
	};
}
