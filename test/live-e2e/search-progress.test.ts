/**
 * Live E2E: a real model answers a TUI question while the search-progress widget is on screen.
 *
 * Gated on AUTORAG_PROGRESS_LIVE=1 plus an OpenRouter credential; the offline suite never calls a
 * model. The librarian, the pi session, Jev routing, retrieval and the model are real; only the
 * terminal is replaced by a recorder, so the test can read exactly what the widget would draw.
 *
 *   AUTORAG_PROGRESS_LIVE=1 AUTORAG_PROGRESS_LIVE_MODEL=qwen/qwen3.7-flash \
 *     bun run vitest run test/live-e2e/search-progress.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, delimiter, join } from "node:path";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { SEARCH_PROGRESS_WIDGET_KEY } from "../../src/agent/search-progress.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const LIVE = process.env.AUTORAG_PROGRESS_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_PROGRESS_LIVE_MODEL ?? "qwen/qwen3.7-flash";

interface WidgetComponent {
	render(width: number): string[];
	dispose?(): void;
}
type WidgetFactory = (
	tui: { requestRender(): void },
	theme: { fg(color: string, text: string): string },
) => WidgetComponent;

/** The developer's own `minsync`, skipping the inert stand-in the test setup prepends to PATH. */
function realMinSync(): string {
	const name = process.platform === "win32" ? "minsync.exe" : "minsync";
	for (const directory of (process.env.PATH ?? "").split(delimiter)) {
		if (directory === "" || basename(directory).startsWith("autorag-fake-minsync-path-")) continue;
		const candidate = join(directory, name);
		if (existsSync(candidate)) return candidate;
	}
	throw new Error("a real `minsync` on PATH is required for the live progress test");
}

let root: string;
let savedEnv: Record<string, string | undefined>;

// The suite's setup files stub Kiwi (an empty library file) and the pi agent dir; the real MinSync would
// fail to load that stub, so this lane drops the Kiwi variables to get the developer's real discovery.
const RESTORED_VARIABLES = ["PI_CODING_AGENT_DIR", "KIWI_LIBRARY_PATH", "KIWI_MODEL_PATH"] as const;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "arag-progress-live-"));
	savedEnv = Object.fromEntries(RESTORED_VARIABLES.map((name) => [name, process.env[name]]));
	delete process.env.KIWI_LIBRARY_PATH;
	delete process.env.KIWI_MODEL_PATH;
});

afterEach(() => {
	for (const name of RESTORED_VARIABLES) {
		const value = savedEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	rmSync(root, { recursive: true, force: true });
});

describe.skipIf(!LIVE)("search progress widget, live model", () => {
	it("is on screen from the moment a question is submitted until the answer is final, and names what ran", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(
			join(docs, "lease.md"),
			"# Seoul office lease\n\nThe Seoul office lease renews on 2027-03-31. Monthly rent is 4,800,000 KRW.\n",
		);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
		const config = resolveConfig({
			flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
			env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel(config);
		const agent = new AutoRAGAgent({
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false, binaryPath: realMinSync() },
			jikji: false,
			// Machine-wide finders would search the developer's real disk, not the fixture corpus.
			everything: false,
			fsearch: false,
			webSearch: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
		});
		// The suite puts an inert `minsync` first on PATH; this lane needs the real one, found on the
		// PATH the developer started with, and a real index to query.
		const refreshed = await agent.refresh(true, { methods: ["parsed", "minsync"] });
		expect(refreshed.minsync?.ok, JSON.stringify(refreshed.minsync)).toBe(true);

		// What the terminal would draw: every frame the widget rendered while the search ran.
		const frames: string[] = [];
		const events: boolean[] = [];
		let current: WidgetComponent | undefined;
		const sample = (): void => {
			if (current !== undefined) frames.push(current.render(100).join("\n"));
		};
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			await hosted.runtime.session.bindExtensions({
				mode: "tui",
				uiContext: {
					notify: () => undefined,
					setStatus: () => undefined,
					setWidget: (key: string, content: unknown) => {
						if (key !== SEARCH_PROGRESS_WIDGET_KEY) return;
						current?.dispose?.();
						current = undefined;
						events.push(content !== undefined);
						if (typeof content === "function") {
							current = (content as WidgetFactory)({ requestRender: sample }, { fg: (_c, text) => text });
							sample();
						}
					},
					confirm: async () => true,
					select: async () => undefined,
					input: async () => undefined,
					onTerminalInput: () => () => undefined,
					setWorkingMessage: () => undefined,
					setWorkingVisible: () => undefined,
					setTitle: () => undefined,
					theme: undefined as never,
				} as unknown as ExtensionUIContext,
			});

			await hosted.runtime.session.prompt("When does the Seoul office lease renew?", { source: "interactive" });
			// pi's input handler returned: the widget is already up, before any retrieval or model token.
			expect(events).toEqual([true]);
			expect(frames[0]).toContain("Searching");

			await agent.waitForIdle();

			expect(events).toEqual([true, false]);
			const seen = frames.join("\n");
			// No Jev here, so no routing or verification-check step: the baseline retrieval and both model
			// phases are what shows, and nothing claims a check that never ran.
			expect(seen).toMatch(/MinSync semantic search — \d+ hits?/u);
			expect(seen).toMatch(/Drafting a first answer/u);
			expect(seen).toMatch(/Verifying the answer against the sources/u);
			expect(seen).not.toMatch(/Routing the question|Checking whether/u);
			// The elapsed time ticked while the search ran, so a stuck run would be visibly stuck.
			expect(new Set(frames.map((frame) => frame.split("\n")[0])).size).toBeGreaterThan(1);

			const answers = hosted.runtime.session.messages.flatMap((message) =>
				message.role === "assistant"
					? message.content.flatMap((block) => (block.type === "text" ? [block.text] : []))
					: [],
			);
			expect(answers.join("\n")).toMatch(/2027/u);
		} finally {
			await hosted.dispose();
		}
	}, 300_000);
});
