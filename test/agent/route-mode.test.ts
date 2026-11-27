import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, type FauxResponseStep, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { DECOMPOSE_QUESTION_ID, QUERY_ROUTE_QUESTION_ID } from "../../src/agent/query-routing.ts";
import {
	availableRouteModes,
	BASE_ROUTE_MODES,
	createRouteModeControls,
	cycleRouteMode,
	DEFAULT_ROUTE_MODE,
	describeRouteMode,
	parseRouteMode,
	ROUTE_MODE_STATUS_KEY,
	ROUTE_MODES,
	type RouteMode,
	RouteModeController,
	routeModeStatusLabel,
	routeOverrideForMode,
} from "../../src/agent/route-mode.ts";

type EventHandler = (event: Record<string, unknown>, ctx: ExtensionContext) => unknown;

interface CommandReg {
	handler: (args: string, ctx: ExtensionContext) => Promise<void>;
	getArgumentCompletions?: (prefix: string) => { value: string; label: string; description?: string }[] | null;
}

function createHarness(
	options: { mode?: RouteMode; selfConfigEnabled?: boolean; answers?: (string | undefined)[] } = {},
) {
	const handlers = new Map<string, EventHandler[]>();
	const commands = new Map<string, CommandReg>();
	const controller = new RouteModeController(options.mode ?? DEFAULT_ROUTE_MODE, {
		selfConfigEnabled: options.selfConfigEnabled,
	});
	const pi = {
		on: (event: string, handler: EventHandler) => {
			handlers.set(event, [...(handlers.get(event) ?? []), handler]);
			return () => undefined;
		},
		registerCommand: (name: string, reg: CommandReg) => {
			commands.set(name, reg);
		},
	} as unknown as ExtensionAPI;

	const answers = [...(options.answers ?? [])];
	const terminalListeners = new Set<(data: string) => { consume?: boolean } | undefined>();
	const statuses: { key: string; text: string | undefined }[] = [];
	const notices: { message: string; level?: string }[] = [];
	const select = vi.fn(async () => answers.shift());
	const ctx = {
		hasUI: true,
		ui: {
			select,
			notify: (message: string, level?: string) => void notices.push({ message, level }),
			setStatus: (key: string, text: string | undefined) => void statuses.push({ key, text }),
			onTerminalInput: (listener: (data: string) => { consume?: boolean } | undefined) => {
				terminalListeners.add(listener);
				return () => terminalListeners.delete(listener);
			},
			theme: { fg: (_color: string, text: string) => text },
		},
	} as unknown as ExtensionContext;

	createRouteModeControls({ controller })(pi);
	const fire = (event: string) => {
		for (const handler of handlers.get(event) ?? []) handler({}, ctx);
	};
	const press = (raw: string) => {
		for (const listener of terminalListeners) {
			const res = listener(raw);
			if (res?.consume) return true;
		}
		return false;
	};
	const runMode = async (args: string) => {
		const cmd = commands.get("mode");
		if (cmd === undefined) throw new Error("/mode command was not registered");
		await cmd.handler(args, ctx);
	};
	const complete = (prefix: string) => commands.get("mode")?.getArgumentCompletions?.(prefix) ?? null;

	return { controller, fire, press, runMode, complete, statuses, notices, select, commands };
}

describe("route-mode helpers", () => {
	it("parses canonical modes and friendly aliases case-insensitively", () => {
		expect(parseRouteMode("auto")).toBe("auto");
		expect(parseRouteMode("AUTO")).toBe("auto");
		expect(parseRouteMode("jev")).toBe("auto");
		expect(parseRouteMode("local")).toBe("local");
		expect(parseRouteMode("find")).toBe("local");
		expect(parseRouteMode("search")).toBe("local");
		expect(parseRouteMode("summarize")).toBe("local");
		expect(parseRouteMode("explore")).toBe("local");
		expect(parseRouteMode("organize")).toBe("organize");
		expect(parseRouteMode("extra")).toBe("extra");
		expect(parseRouteMode("direct")).toBe("extra");
		expect(parseRouteMode("config")).toBe("config");
		expect(parseRouteMode("automate")).toBe("config");
		expect(parseRouteMode("")).toBeUndefined();
		expect(parseRouteMode("bogus")).toBeUndefined();
		expect(parseRouteMode(undefined)).toBeUndefined();
	});

	it("maps auto to undefined routeOverride and explicit routes to themselves", () => {
		expect(routeOverrideForMode("auto")).toBeUndefined();
		expect(routeOverrideForMode("local")).toBe("local");
		expect(routeOverrideForMode("organize")).toBe("organize");
		expect(routeOverrideForMode("extra")).toBe("extra");
		expect(routeOverrideForMode("config")).toBe("config");
	});

	it("cycles through all modes in order and omits config when selfConfigEnabled is false", () => {
		expect(availableRouteModes(true)).toEqual(ROUTE_MODES);
		expect(availableRouteModes(false)).toEqual(BASE_ROUTE_MODES);
		expect(cycleRouteMode("auto", true)).toBe("local");
		expect(cycleRouteMode("local", true)).toBe("organize");
		expect(cycleRouteMode("organize", true)).toBe("extra");
		expect(cycleRouteMode("extra", true)).toBe("config");
		expect(cycleRouteMode("config", true)).toBe("auto");
		expect(cycleRouteMode("extra", false)).toBe("auto");
	});

	it("provides non-empty labels and descriptions for every mode", () => {
		for (const mode of ROUTE_MODES) {
			expect(routeModeStatusLabel(mode)).toContain(mode);
			expect(describeRouteMode(mode).length).toBeGreaterThan(5);
		}
	});
});

describe("createRouteModeControls (TUI Ctrl+P and /mode)", () => {
	it("starts in auto, shows footer status on session_start, and cycles modes with Ctrl+P", () => {
		const h = createHarness();
		expect(h.controller.mode).toBe("auto");
		expect(h.controller.routeOverride).toBeUndefined();

		h.fire("session_start");
		expect(h.statuses.at(-1)).toEqual({ key: ROUTE_MODE_STATUS_KEY, text: "mode: auto" });

		// Ctrl+P is raw ASCII 0x10 (\x10)
		expect(h.press("\x10")).toBe(true);
		expect(h.controller.mode).toBe("local");
		expect(h.controller.routeOverride).toBe("local");
		expect(h.statuses.at(-1)).toEqual({ key: ROUTE_MODE_STATUS_KEY, text: "mode: local" });
		expect(h.notices.at(-1)?.message).toContain("local");

		expect(h.press("\x10")).toBe(true);
		expect(h.controller.mode).toBe("organize");
		expect(h.press("\x10")).toBe(true);
		expect(h.controller.mode).toBe("extra");
		expect(h.press("\x10")).toBe(true);
		expect(h.controller.mode).toBe("config");
		expect(h.press("\x10")).toBe(true);
		expect(h.controller.mode).toBe("auto");

		// Unrelated keys are not consumed
		expect(h.press("a")).toBe(false);
		expect(h.controller.mode).toBe("auto");

		// Unsubscribes on shutdown
		h.fire("session_shutdown");
		expect(h.press("\x10")).toBe(false);
	});

	it("supports /mode <mode>, aliases, interactive picker, completions, and rejects invalid modes", async () => {
		const h = createHarness({ answers: ["organize - organize files"] });
		h.fire("session_start");

		// Argument completions
		expect(h.complete("")?.map((item) => item.value)).toEqual([...ROUTE_MODES]);
		expect(h.complete("or")?.map((item) => item.value)).toEqual(["organize"]);
		expect(h.complete("zzz")).toBeNull();

		// Direct switch + alias
		await h.runMode("organize");
		expect(h.controller.mode).toBe("organize");
		await h.runMode("find");
		expect(h.controller.mode).toBe("local");
		await h.runMode("automate");
		expect(h.controller.mode).toBe("config");
		await h.runMode("auto");
		expect(h.controller.mode).toBe("auto");

		// Invalid mode warns and does not change state
		await h.runMode("invalid-mode");
		expect(h.controller.mode).toBe("auto");
		expect(h.notices.at(-1)?.level).toBe("warning");

		// No argument opens interactive selector
		await h.runMode("");
		expect(h.select).toHaveBeenCalledTimes(1);
		expect(h.controller.mode).toBe("organize");
	});

	it("omits config from completions, cycling, and /mode when selfConfigEnabled is false", async () => {
		const h = createHarness({ selfConfigEnabled: false });
		h.fire("session_start");
		expect(h.complete("")?.map((item) => item.value)).toEqual([...BASE_ROUTE_MODES]);
		await h.runMode("config");
		expect(h.controller.mode).toBe("auto");
		expect(h.notices.at(-1)?.level).toBe("warning");
	});
});

describe("AutoRAGAgent routeModeController integration", () => {
	let root: string;
	let docs: string;
	let registrations: FauxProviderRegistration[];

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-route-mode-agent-"));
		docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "Project deadline is Friday.", "utf8");
		registrations = [];
	});

	afterEach(() => {
		for (const reg of registrations) reg.unregister();
		rmSync(root, { recursive: true, force: true });
	});

	function fauxModel(...steps: FauxResponseStep[]) {
		const reg = registerFauxProvider({ api: `faux-rm-${randomUUID()}`, models: [{ id: "faux-model" }] });
		reg.setResponses(steps);
		registrations.push(reg);
		return reg.getModel();
	}

	it("skips Jev route classification when routeMode is set to organize, while auto still calls Jev", async () => {
		const askedQuestionIds: string[][] = [];
		const backend: JevBackend = {
			name: "recording",
			async judge(request) {
				askedQuestionIds.push(request.questions.map((q) => q.id));
				return {
					answers: request.questions.map((q) =>
						q.id === QUERY_ROUTE_QUESTION_ID
							? { answer: "extra", distribution: { local: 0.05, organize: 0.05, extra: 0.9 } }
							: q.id === DECOMPOSE_QUESTION_ID
								? { answer: 0.1 }
								: { answer: 0.9 },
					),
				};
			},
		};

		// 1. Explicit organize mode via routeModeController bypasses QUERY_ROUTE_QUESTION_ID
		const controller = new RouteModeController("organize");
		const organizeAgent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "mem1.json"),
			model: fauxModel(fauxAssistantMessage("Organized files.", { stopReason: "stop" })),
			jev: { backend },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			routeModeController: controller,
		});
		const organizeRes = await organizeAgent.searchDocuments("tidy my docs folder");
		expect(organizeRes.answer).toContain("Organized files.");
		const routedDiag = organizeRes.diagnostics?.find((d) => d.code === "query-routed");
		expect(routedDiag?.source).toBe("route-override");
		expect(routedDiag?.message).toContain("Caller explicitly routed the question to organize");
		// After the decision-model health check, no QUERY_ROUTE_QUESTION_ID should be sent
		const routeCalls = askedQuestionIds.filter((ids) => ids.includes(QUERY_ROUTE_QUESTION_ID));
		expect(routeCalls).toHaveLength(0);

		// 2. Switching controller back to auto re-enables Jev routing
		askedQuestionIds.length = 0;
		controller.set("auto");
		const autoAgent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "mem2.json"),
			model: fauxModel(fauxAssistantMessage("2 + 2 = 4.", { stopReason: "stop" })),
			jev: { backend },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			routeModeController: controller,
		});
		const autoRes = await autoAgent.searchDocuments("what is 2 + 2?");
		const autoDiag = autoRes.diagnostics?.find((d) => d.code === "query-routed");
		expect(autoDiag?.source).toBe("jev");
		expect(askedQuestionIds.some((ids) => ids.includes(QUERY_ROUTE_QUESTION_ID))).toBe(true);
	});

	it("per-call options.routeOverride takes precedence over agent routeModeController", async () => {
		const controller = new RouteModeController("organize");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "mem3.json"),
			model: fauxModel(fauxAssistantMessage("Direct extra answer.", { stopReason: "stop" })),
			jev: { backend: new MockBackend({}) },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			routeModeController: controller,
		});
		const res = await agent.searchDocuments("say hi", { routeOverride: "extra" });
		const diag = res.diagnostics?.find((d) => d.code === "query-routed");
		expect(diag?.source).toBe("route-override");
		expect(diag?.message).toContain("Caller explicitly routed the question to extra");
	});

	it("wires /mode command and Ctrl+P into a real interactive Pi runtime and applies mode changes to queries", async () => {
		const askedQuestionIds: string[][] = [];
		const backend: JevBackend = {
			name: "recording",
			async judge(request) {
				askedQuestionIds.push(request.questions.map((q) => q.id));
				return {
					answers: request.questions.map((q) =>
						q.id === QUERY_ROUTE_QUESTION_ID
							? { answer: "local", distribution: { local: 0.9, organize: 0.05, extra: 0.05 } }
							: q.id === DECOMPOSE_QUESTION_ID
								? { answer: 0.1 }
								: { answer: 0.9 },
					),
				};
			},
		};
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "mem-tui.json"),
			piAgentDir: join(root, "pi-agent"),
			piSessionDir: join(root, "pi-sessions"),
			model: fauxModel(
				fauxAssistantMessage("Organized in TUI.", { stopReason: "stop" }),
				fauxAssistantMessage("Direct extra turn in TUI.", { stopReason: "stop" }),
			),
			jev: { backend },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
		});
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			const session = hosted.runtime.session;
			const commands = session.extensionRunner.getRegisteredCommands();
			expect(commands.some((cmd) => cmd.name === "mode")).toBe(true);
			expect(agent.routeModeControl.mode).toBe("auto");

			// Switch to organize via routeModeControl (same controller /mode and Ctrl+P mutate)
			agent.routeModeControl.set("organize");
			expect(agent.routeModeControl.mode).toBe("organize");
			const first = await agent.searchDocuments("tidy my files");
			expect(first.diagnostics?.find((d) => d.code === "query-routed")).toMatchObject({
				source: "route-override",
			});
			expect(first.diagnostics?.find((d) => d.code === "query-routed")?.message).toContain(
				"Caller explicitly routed the question to organize",
			);
			expect(askedQuestionIds.filter((ids) => ids.includes(QUERY_ROUTE_QUESTION_ID))).toHaveLength(0);

			// Second turn in the same interactive session with explicit mode switched to extra still routes explicitly
			agent.routeModeControl.set("extra");
			const second = await agent.searchDocuments("just explain something");
			expect(second.diagnostics?.find((d) => d.code === "query-routed")).toMatchObject({
				source: "route-override",
			});
			expect(second.diagnostics?.find((d) => d.code === "query-routed")?.message).toContain(
				"Caller explicitly routed the question to extra",
			);
			expect(askedQuestionIds.filter((ids) => ids.includes(QUERY_ROUTE_QUESTION_ID))).toHaveLength(0);
		} finally {
			await hosted.dispose();
		}
	});
});
