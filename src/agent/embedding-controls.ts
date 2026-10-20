import type { ExtensionAPI, ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { ensureMinSyncDaemon, stopMinSyncDaemon } from "../embedding/daemon.ts";
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
	return `embedding: off ○ (${toggleKey} starts minsync serve)`;
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

		pi.registerShortcut(EMBEDDING_TOGGLE_KEY, {
			description: "Start or stop the local embedding daemon (minsync serve)",
			handler: async (ctx) => {
				const target = ui(ctx);
				if (options.remote) {
					target.notify("Embedding uses a remote endpoint; there is no local daemon to toggle.", "info");
					return;
				}
				if (busy) return;
				busy = true;
				try {
					const stopped = await stopMinSyncDaemon();
					if (stopped !== undefined) {
						target.notify(`Stopped minsync serve (pid ${stopped.pid ?? "?"}).`, "info");
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
			},
		});
	};
}
