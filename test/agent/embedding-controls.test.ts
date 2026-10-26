import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	createEmbeddingControls,
	EMBEDDING_STATUS_KEY,
	EMBEDDING_TOGGLE_KEY,
	embeddingStatusText,
} from "../../src/agent/embedding-controls.ts";
import { probeMinSyncDaemon } from "../../src/embedding/daemon.ts";
import type { EmbedderHealth, ManagedEmbedder } from "../../src/embedding/embedder.ts";
import { createManagedEmbedder } from "../../src/embedding/managed.ts";
import { writeFakeMinSyncServe } from "../helpers/fake-minsync-serve.ts";

describe("embeddingStatusText", () => {
	it("names the backend and whether it answers, and tells the user the key when nothing runs", () => {
		const text = (health: EmbedderHealth) => embeddingStatusText(health);
		expect(text({ backend: "daemon", alive: true })).toBe("embedding: daemon ● up");
		expect(text({ backend: "in-process", alive: true })).toBe("embedding: in-process ●");
		expect(text({ backend: "remote", alive: true })).toBe("embedding: remote ● up");
		expect(text({ backend: "remote", alive: false, detail: "HTTP 503" })).toBe("embedding: remote ○ down (HTTP 503)");
		expect(text({ backend: "none", alive: false })).toBe(
			`embedding: off ○ (/embedding on, or ${EMBEDDING_TOGGLE_KEY}, starts minsync serve)`,
		);
	});
});

/** The part of pi's registered command the tests drive. */
interface RegisteredCommandShape {
	readonly handler: (args: string, ctx: ExtensionContext) => Promise<void>;
	readonly getArgumentCompletions?: (
		prefix: string,
	) => { value: string }[] | null | Promise<{ value: string }[] | null>;
}

interface Harness {
	readonly statuses: (string | undefined)[];
	readonly notes: { readonly message: string; readonly type: string | undefined }[];
	readonly fire: (event: "session_start" | "session_shutdown") => void;
	readonly press: () => Promise<void>;
	readonly run: (args: string) => Promise<void>;
	readonly complete: (prefix: string) => Promise<{ value: string }[] | null>;
	readonly shortcuts: string[];
}

/** A pi that records what the extension registers and lets the test drive its events and its key. */
function install(
	embedder: ManagedEmbedder,
	options: { remote: boolean; resolveBinary: () => Promise<string> },
): Harness {
	const handlers = new Map<string, (event: unknown, ctx: ExtensionContext) => void>();
	let shortcut: ((ctx: ExtensionContext) => Promise<void>) | undefined;
	let command: RegisteredCommandShape | undefined;
	const shortcuts: string[] = [];
	const statuses: (string | undefined)[] = [];
	const notes: { message: string; type: string | undefined }[] = [];
	const ctx = {
		hasUI: true,
		ui: {
			setStatus: (_key: string, text: string | undefined) => statuses.push(text),
			notify: (message: string, type?: string) => notes.push({ message, type }),
		},
	} as unknown as ExtensionContext;
	const pi = {
		on: (event: string, handler: (event: unknown, ctx: ExtensionContext) => void) => handlers.set(event, handler),
		registerShortcut: (key: string, registration: { handler: (ctx: ExtensionContext) => Promise<void> }) => {
			shortcuts.push(key);
			shortcut = registration.handler;
		},
		registerCommand: (name: string, registration: RegisteredCommandShape) => {
			if (name === "embedding") command = registration;
		},
	} as unknown as ExtensionAPI;
	createEmbeddingControls({ embedder, ...options })(pi);
	return {
		statuses,
		notes,
		shortcuts,
		fire: (event) => handlers.get(event)?.({}, ctx),
		press: async () => shortcut?.(ctx),
		run: async (args) => command?.handler(args, ctx),
		complete: async (prefix) => (await command?.getArgumentCompletions?.(prefix)) ?? null,
	};
}

describe("createEmbeddingControls", () => {
	let home: string;
	let bin: string;
	beforeEach(() => {
		home = mkdtempSync(join(tmpdir(), "autorag-controls-home-"));
		bin = mkdtempSync(join(tmpdir(), "autorag-controls-bin-"));
		vi.stubEnv("HOME", home);
	});
	afterEach(() => {
		vi.unstubAllEnvs();
		rmSync(home, { recursive: true, force: true });
		rmSync(bin, { recursive: true, force: true });
	});

	it("registers the toggle key and shows the probed state as soon as the session starts", async () => {
		const embedder = createManagedEmbedder({
			home,
			inProcess: async () => {
				throw new Error("no addon");
			},
		});
		const harness = install(embedder, { remote: false, resolveBinary: async () => join(bin, "minsync") });
		expect(harness.shortcuts).toEqual([EMBEDDING_TOGGLE_KEY]);
		harness.fire("session_start");
		await vi.waitFor(() =>
			expect(harness.statuses).toEqual([embeddingStatusText({ backend: "none", alive: false })]),
		);
		harness.fire("session_shutdown");
	});

	it("does nothing to a remote endpoint's key press except explain, and shows the remote's /health", async () => {
		const embedder = createManagedEmbedder({
			remote: { baseUrl: "https://emb.example.com", model: "m", dimension: 2, protocol: "openai" },
			fetchImpl: (async () => new Response("{}", { status: 200 })) as unknown as typeof fetch,
		});
		let resolved = false;
		const harness = install(embedder, {
			remote: true,
			resolveBinary: async () => {
				resolved = true;
				return "unused";
			},
		});
		harness.fire("session_start");
		await vi.waitFor(() => expect(harness.statuses).toEqual(["embedding: remote ● up"]));
		await harness.press();
		expect(harness.notes).toEqual([{ message: expect.stringContaining("remote endpoint"), type: "info" }]);
		expect(resolved).toBe(false);
		harness.fire("session_shutdown");
	});

	it("starts the daemon on the first press and stops it on the second, refreshing the footer each time", async () => {
		const embedder = createManagedEmbedder({
			home,
			inProcess: async () => {
				throw new Error("no addon");
			},
		});
		const binaryPath = writeFakeMinSyncServe(bin, { dimension: 4 });
		const harness = install(embedder, { remote: false, resolveBinary: async () => binaryPath });

		await harness.press();
		expect(await probeMinSyncDaemon({ home })).toBeDefined();
		expect(harness.notes[0]).toMatchObject({
			message: expect.stringContaining("Started minsync serve"),
			type: "info",
		});
		expect(harness.statuses.at(-1)).toBe("embedding: daemon ● up");

		await harness.press();
		expect(await probeMinSyncDaemon({ home })).toBeUndefined();
		expect(harness.notes[1]).toMatchObject({
			message: expect.stringContaining("Stopped minsync serve"),
			type: "info",
		});
		expect(harness.statuses.at(-1)).toBe(embeddingStatusText({ backend: "none", alive: false }));
	}, 60_000);

	it("shows the real reason as an error when the daemon cannot be started, and leaves the footer truthful", async () => {
		const embedder = createManagedEmbedder({
			home,
			inProcess: async () => {
				throw new Error("no addon");
			},
		});
		const harness = install(embedder, {
			remote: false,
			resolveBinary: async () => writeFakeMinSyncServe(bin, { failWith: "cannot load model: no cached weights" }),
		});
		await harness.press();
		expect(harness.notes).toEqual([{ message: expect.stringContaining("no cached weights"), type: "error" }]);
		expect(harness.statuses.at(-1)).toBe(embeddingStatusText({ backend: "none", alive: false }));
	}, 60_000);

	it("never touches pi's ctx after its session ended: an in-flight probe is dropped and the poll stops", async () => {
		vi.useFakeTimers();
		try {
			// tsconfig lib is ES2022, which predates Promise.withResolvers.
			let release: () => void = () => undefined;
			const released = new Promise<void>((resolve) => {
				release = resolve;
			});
			let healthCalls = 0;
			const embedder: ManagedEmbedder = {
				identity: async () => ({ provider: "minsync", model: "m", dimension: 1 }),
				embed: async () => [],
				health: async () => {
					healthCalls += 1;
					await released;
					return { backend: "daemon", alive: true };
				},
				describeEndpoint: async () => undefined,
				close: async () => undefined,
			};
			let ended = false;
			const usesAfterEnd: string[] = [];
			const ctx = {
				hasUI: true,
				ui: {
					// pi throws on any use of a ctx whose session was replaced; record the attempt first.
					setStatus: (_key: string, text: string | undefined) => {
						if (ended) {
							usesAfterEnd.push(String(text));
							throw new Error("This extension ctx is stale after session replacement or reload.");
						}
					},
					notify: () => undefined,
				},
			} as unknown as ExtensionContext;
			const handlers = new Map<string, (event: unknown, ctx: ExtensionContext) => void>();
			const pi = {
				on: (event: string, handler: (event: unknown, ctx: ExtensionContext) => void) =>
					handlers.set(event, handler),
				registerShortcut: () => undefined,
				registerCommand: () => undefined,
			} as unknown as ExtensionAPI;
			createEmbeddingControls({ embedder, remote: false, resolveBinary: async () => "unused" })(pi);

			handlers.get("session_start")?.({}, ctx);
			expect(healthCalls).toBe(1);
			handlers.get("session_shutdown")?.({}, ctx);
			ended = true;
			release();
			await vi.advanceTimersByTimeAsync(0);
			expect(usesAfterEnd).toEqual([]);

			// The poll belongs to the ended session: a minute later it has not probed again.
			await vi.advanceTimersByTimeAsync(60_000);
			expect(healthCalls).toBe(1);
		} finally {
			vi.useRealTimers();
		}
	});

	describe("/embedding", () => {
		function localEmbedder(): ManagedEmbedder {
			return createManagedEmbedder({
				home,
				inProcess: async () => {
					throw new Error("no addon");
				},
			});
		}

		it("starts with `on`, is idempotent, stops with `off`, and says so when there is nothing to do", async () => {
			const harness = install(localEmbedder(), {
				remote: false,
				resolveBinary: async () => writeFakeMinSyncServe(bin, { dimension: 4 }),
			});
			await harness.run("off");
			expect(harness.notes.at(-1)?.message).toBe("minsync serve is not running.");

			await harness.run("on");
			expect(harness.notes.at(-1)?.message).toContain("Started minsync serve");
			expect(await probeMinSyncDaemon({ home })).toBeDefined();

			await harness.run("on");
			expect(harness.notes.at(-1)?.message).toContain("already running");

			await harness.run("off");
			expect(harness.notes.at(-1)?.message).toContain("Stopped minsync serve");
			expect(await probeMinSyncDaemon({ home })).toBeUndefined();
		}, 60_000);

		it("toggles with no argument and with `toggle`, exactly like the key", async () => {
			const harness = install(localEmbedder(), {
				remote: false,
				resolveBinary: async () => writeFakeMinSyncServe(bin, { dimension: 4 }),
			});
			await harness.run("");
			expect(await probeMinSyncDaemon({ home })).toBeDefined();
			await harness.run("toggle");
			expect(await probeMinSyncDaemon({ home })).toBeUndefined();
		}, 60_000);

		it("reports the backend with `status` without starting or stopping anything", async () => {
			const harness = install(localEmbedder(), { remote: false, resolveBinary: async () => "unused" });
			await harness.run("status");
			expect(harness.notes.at(-1)).toMatchObject({
				type: "warning",
				message: expect.stringContaining("/embedding on"),
			});
			expect(await probeMinSyncDaemon({ home })).toBeUndefined();
		});

		it("rejects an unknown argument instead of toggling", async () => {
			const harness = install(localEmbedder(), { remote: false, resolveBinary: async () => "unused" });
			await harness.run("restart");
			expect(harness.notes).toEqual([
				{ message: 'Unknown argument "restart". Use /embedding [on|off|toggle|status].', type: "warning" },
			]);
			expect(await probeMinSyncDaemon({ home })).toBeUndefined();
		});

		it("only explains with a remote endpoint, but `status` still reports its /health", async () => {
			const embedder = createManagedEmbedder({
				remote: { baseUrl: "https://emb.example.com", model: "m", dimension: 2, protocol: "openai" },
				fetchImpl: (async () => new Response("{}", { status: 200 })) as unknown as typeof fetch,
			});
			const harness = install(embedder, { remote: true, resolveBinary: async () => "unused" });
			await harness.run("on");
			expect(harness.notes.at(-1)?.message).toContain("remote endpoint");
			await harness.run("status");
			expect(harness.notes.at(-1)).toMatchObject({ type: "info", message: expect.stringContaining("remote") });
		});

		it("completes its arguments", async () => {
			const harness = install(localEmbedder(), { remote: false, resolveBinary: async () => "unused" });
			expect((await harness.complete(""))?.map((item) => item.value)).toEqual(["on", "off", "toggle", "status"]);
			expect((await harness.complete("o"))?.map((item) => item.value)).toEqual(["on", "off"]);
			expect(await harness.complete("zzz")).toBeNull();
		});
	});

	it("keeps a footer slot of its own", () => {
		expect(EMBEDDING_STATUS_KEY).toBe("autorag-embedding");
	});
});
