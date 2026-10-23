import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessage, SimpleStreamOptions, SystemMessage, TranscriptContext } from "@earendil-works/pi-ai";
import {
	type FauxProviderRegistration,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type Automation, parseAutomations } from "../../src/automation/config.ts";
import { AGENT_RUN_CONCURRENCY, runRule } from "../../src/automation/dispatch.ts";
import { buildAutomationProfile, eventsPrompt } from "../../src/automation/run-agent.ts";

interface Seen {
	systemPrompt: string;
	userText: string;
	toolNames: string[];
}

let root: string;
let home: string;
let watched: string;
let agentDir: string;
const unregister: Array<() => void> = [];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-run-agent-"));
	home = join(root, "home");
	watched = join(root, "inbox");
	agentDir = join(root, "agent");
	for (const dir of [home, watched, agentDir]) mkdirSync(dir, { recursive: true });
});

afterEach(() => {
	for (const fn of unregister.splice(0)) fn();
	rmSync(root, { recursive: true, force: true });
});

function rule(extra: Record<string, unknown> = {}): Automation {
	const [automation] = parseAutomations(
		{
			automations: [
				{
					id: "triage",
					paths: [watched],
					include: ["**/*.pdf"],
					agent: { profile: "file-triage", task: "Summarize each file." },
					...extra,
				},
			],
		},
		{ HOME: root },
	);
	return automation as Automation;
}

function text(message: SystemMessage): string {
	const content =
		typeof message.content === "string" ? message.content : message.content.map((block) => block.text).join("\n");
	const sections = Object.values(message.sections ?? {}).filter((value): value is string => typeof value === "string");
	return [content, ...sections].join("\n");
}

function observe(context: TranscriptContext): Seen {
	const tools = new Set<string>();
	const prompt: string[] = [];
	const user: string[] = [];
	for (const message of context.messages) {
		if (message.role === "user") {
			user.push(typeof message.content === "string" ? message.content : JSON.stringify(message.content));
		}
		if (message.role !== "system") continue;
		prompt.push(text(message));
		for (const tool of message.toolsAdded ?? []) tools.add(tool.name);
		for (const tool of message.toolsRemoved ?? []) tools.delete(tool.name);
	}
	return { systemPrompt: prompt.join("\n"), userText: user.join("\n"), toolNames: [...tools].sort() };
}

function model(steps: Array<(seen: Seen) => AssistantMessage>): {
	seen: Seen[];
	registration: FauxProviderRegistration;
} {
	const registration = registerFauxProvider({
		api: `faux-run-agent-${Math.random().toString(36).slice(2)}`,
		provider: "faux-run-agent",
		models: [{ id: "faux-model", reasoning: true }],
	});
	unregister.push(() => registration.unregister());
	const seen: Seen[] = [];
	registration.setResponses(
		steps.map((step) => (context: TranscriptContext, _options: SimpleStreamOptions | undefined) => {
			const request = observe(context);
			seen.push(request);
			return step(request);
		}),
	);
	return { seen, registration };
}

describe("buildAutomationProfile", () => {
	it("appends the loop guard to a read-only profile", () => {
		const profile = buildAutomationProfile(rule());
		expect(profile.systemPrompt).toContain("Summarize each file.");
		expect(profile.systemPrompt).toContain("Automation loop guard");
		expect(profile.systemPrompt).toContain(watched);
		expect(profile.systemPrompt).toContain(".pdf");
	});

	it("appends the loop guard to a profile that may write, which is where it matters", () => {
		const profile = buildAutomationProfile(
			rule({ agent: { profile: "file-organizer", task: "Sort files by year." } }),
		);
		expect(profile.tools).toEqual(expect.arrayContaining(["bash", "edit", "write"]));
		expect(profile.systemPrompt).toContain("Automation loop guard");
		expect(profile.systemPrompt).toContain("Triggering this automation is forbidden");
	});

	it("gives the read-only profile no way to change a file", () => {
		expect(buildAutomationProfile(rule()).tools).toEqual(["read", "grep", "find", "ls"]);
	});
});

describe("eventsPrompt", () => {
	it("lists one line per change and says how many were left out past the cap", () => {
		const events = Array.from({ length: 103 }, (_, index) => ({ type: "create" as const, path: `/w/f${index}.pdf` }));
		const prompt = eventsPrompt(events);
		expect(prompt).toContain("103 file change(s)");
		expect(prompt).toContain("create: /w/f0.pdf");
		expect(prompt).toContain("create: /w/f99.pdf");
		expect(prompt).not.toContain("f100.pdf");
		expect(prompt).toContain("(3 more change(s) not listed)");
	});
});

describe("runRule for an agent rule", () => {
	it("sends the loop guard, the changed files, and only the profile's tools; logs the final reply", async () => {
		const { seen, registration } = model([() => fauxAssistantMessage("Both files are invoices.")]);
		const record = await runRule(
			rule(),
			[
				{ type: "create", path: join(watched, "a.pdf") },
				{ type: "update", path: join(watched, "b.pdf") },
			],
			{ home, agentDir, env: { HOME: root }, resolveModel: async () => ({ model: registration.getModel() }) },
		);

		expect(seen).toHaveLength(1);
		expect(seen[0]?.systemPrompt).toContain("Automation loop guard");
		expect(seen[0]?.toolNames).toEqual(["find", "grep", "ls", "read"]);
		expect(seen[0]?.userText).toContain(`create: ${join(watched, "a.pdf")}`);
		expect(seen[0]?.userText).toContain(`update: ${join(watched, "b.pdf")}`);
		expect(record).toMatchObject({
			id: "triage",
			eventCount: 2,
			exitCode: 0,
			timedOut: false,
			stdout: "Both files are invoices.",
		});
		const logged = readFileSync(join(home, "automations", "logs", "triage.jsonl"), "utf8")
			.trim()
			.split("\n");
		expect(JSON.parse(logged.at(-1) ?? "{}")).toMatchObject({ id: "triage", stdout: "Both files are invoices." });
	});

	it("reports a model failure as a failed run instead of a silent success", async () => {
		const { registration } = model([
			() => ({ ...fauxAssistantMessage(""), stopReason: "error", errorMessage: "provider exploded" }),
		]);
		const record = await runRule(rule(), [{ type: "create", path: join(watched, "a.pdf") }], {
			home,
			agentDir,
			env: { HOME: root },
			resolveModel: async () => ({ model: registration.getModel() }),
		});
		expect(record.exitCode).toBe(1);
		expect(record.error).toBe("provider exploded");
	});

	it("fails the run, not the daemon, when the model cannot be resolved", async () => {
		const record = await runRule(rule(), [{ type: "create", path: join(watched, "a.pdf") }], {
			home,
			agentDir,
			env: { HOME: root },
			resolveModel: async () => {
				throw new Error("no model configured");
			},
		});
		expect(record).toMatchObject({ exitCode: 1, error: "no model configured" });
	});

	it("runs a tool the model calls without touching retrieval memory or the user's real home", async () => {
		const target = join(watched, "a.pdf");
		writeFileSync(target, "x");
		const { registration } = model([
			() => fauxAssistantMessage(fauxToolCall("read", { path: target })),
			() => fauxAssistantMessage("read it"),
		]);
		const record = await runRule(rule(), [{ type: "create", path: target }], {
			home,
			agentDir,
			env: { HOME: root },
			resolveModel: async () => ({ model: registration.getModel() }),
		});
		expect(record.exitCode).toBe(0);
		expect(existsSync(join(home, "memory.json"))).toBe(false);
	});
});

describe("agent run concurrency", () => {
	it("starts the second agent run only after the first finished", async () => {
		expect(AGENT_RUN_CONCURRENCY).toBe(1);
		const { registration } = model([() => fauxAssistantMessage("first"), () => fauxAssistantMessage("second")]);
		const started: number[] = [];
		let release: () => void = () => {};
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		let resolved = 0;
		const options = {
			home,
			agentDir,
			env: { HOME: root } as NodeJS.ProcessEnv,
			resolveModel: async () => {
				resolved += 1;
				started.push(resolved);
				// The first run holds here, inside its slot, until the test lets it go.
				if (resolved === 1) await gate;
				return { model: registration.getModel() };
			},
		};
		const events = [{ type: "create" as const, path: join(watched, "a.pdf") }];
		const first = runRule(rule(), events, options);
		const second = runRule(rule(), events, options);
		// Drain every microtask the second run could use to start; it must still be waiting for the slot.
		await new Promise<void>((resolve) => setImmediate(resolve));
		expect(started).toEqual([1]);
		release();
		const [one, two] = await Promise.all([first, second]);
		expect(started).toEqual([1, 2]);
		expect([one.stdout, two.stdout]).toEqual(["first", "second"]);
	});
});
