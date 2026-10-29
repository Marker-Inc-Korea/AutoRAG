/**
 * Live E2E: a real automation daemon, a real file event, a real hosted model.
 *
 * Gated on AUTORAG_AUTOMATION_AGENT_LIVE=1 so the offline suite and CI (no stored `/login`
 * credential) never touch the network. Run explicitly after `/login`:
 *
 *   AUTORAG_AUTOMATION_AGENT_LIVE=1 [AUTORAG_DAZZI_AGENT_DIR=~/.pi/agent] \
 *   [AUTORAG_AUTOMATION_AGENT_MODEL=deepseek/deepseek-v4.1-flash] \
 *   bunx vitest run test/live-e2e/automation-agent.test.ts
 *
 * Proves, with a real parcel watcher and a real model:
 *  - a file dropped into a watched folder starts the agent and its final reply lands in the run log;
 *  - a read-only profile never changes the watched folder;
 *  - a profile that may write keeps its output out of the watched folder, and the rule is not
 *    re-triggered by what it did (the run count stays at one).
 */
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type AutomationDaemonHandle, startAutomationDaemon } from "../../src/automation/daemon.ts";
import { runRule } from "../../src/automation/dispatch.ts";
import { type AutomationRunRecord, automationLogPath } from "../../src/automation/run.ts";
import { PROVIDER_ID } from "../../src/cloud/config.ts";
import type { ProfileModel } from "../../src/profiles/agent-factory.ts";
import { openPlanRuntime } from "../../src/profiles/session-model.ts";

const LIVE = process.env.AUTORAG_AUTOMATION_AGENT_LIVE === "1";
// The vitest setup isolates PI_CODING_AGENT_DIR, so the real login is read explicitly.
const AGENT_DIR = process.env.AUTORAG_DAZZI_AGENT_DIR ?? join(homedir(), ".pi", "agent");
const MODEL_ID = process.env.AUTORAG_AUTOMATION_AGENT_MODEL ?? "deepseek/deepseek-v4.1-flash";

let root: string;
let home: string;
let inbox: string;
let outbox: string;
let daemon: AutomationDaemonHandle | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-automation-live-"));
	home = join(root, "home");
	inbox = join(root, "inbox");
	outbox = join(root, "outbox");
	for (const dir of [home, inbox, outbox]) mkdirSync(dir, { recursive: true });
});

afterEach(async () => {
	await daemon?.stop();
	daemon = undefined;
	rmSync(root, { recursive: true, force: true });
});

function writeRules(rules: unknown[]): void {
	writeFileSync(join(home, "automations.json"), JSON.stringify({ automations: rules }));
}

async function resolveModel(): Promise<ProfileModel> {
	const runtime = await openPlanRuntime(AGENT_DIR, {});
	const model = runtime.getModel(PROVIDER_ID, MODEL_ID);
	if (model === undefined)
		throw new Error(`${PROVIDER_ID}/${MODEL_ID} is not in the stored catalog; run /login first`);
	return { model };
}

interface Observed {
	readonly runs: AutomationRunRecord[];
	readonly errors: string[];
	/** Settles with the first finished run. */
	readonly first: Promise<AutomationRunRecord>;
}

/** Starts the daemon on the temp home with the real model and the real parcel watcher. */
async function startLive(): Promise<Observed> {
	const runs: AutomationRunRecord[] = [];
	const errors: string[] = [];
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	let settle: (record: AutomationRunRecord) => void = () => {};
	const first = new Promise<AutomationRunRecord>((resolve) => {
		settle = resolve;
	});
	const env = { ...process.env, HOME: root };
	daemon = await startAutomationDaemon({
		home,
		env,
		onRun: (_rule, record) => {
			runs.push(record);
			settle(record);
		},
		onError: (message) => errors.push(message),
		runner: (rule, events) => runRule(rule, events, { home, env, agentDir: AGENT_DIR, resolveModel }),
	});
	return { runs, errors, first };
}

describe.skipIf(!LIVE)("automation agent live e2e (real daemon, real model)", () => {
	it("starts a read-only agent on a new file and leaves the watched folder untouched", async () => {
		writeRules([
			{
				id: "triage",
				paths: [inbox],
				include: ["**/*.txt"],
				debounceMs: 500,
				timeoutMs: 180_000,
				agent: {
					profile: "file-triage",
					task: "Read each changed file and reply with one sentence naming what it is about. Include the word TRIAGED.",
				},
			},
		]);
		const { runs, errors, first } = await startLive();

		const file = join(inbox, "invoice.txt");
		writeFileSync(file, "Invoice 2026-114 from Sunrise Printing for 40 conference badges, total 480000 KRW.");
		const record = await first;

		expect(errors).toEqual([]);
		expect(record).toMatchObject({ id: "triage", eventCount: 1, exitCode: 0, timedOut: false });
		expect(record.stdout).toMatch(/TRIAGED/i);
		expect(readdirSync(inbox)).toEqual(["invoice.txt"]);
		expect(readFileSync(file, "utf8")).toContain("Sunrise Printing");
		expect(runs).toHaveLength(1);
		const logged = readFileSync(automationLogPath(home, "triage"), "utf8").trim().split("\n");
		expect(JSON.parse(logged.at(-1) ?? "{}")).toMatchObject({ id: "triage", exitCode: 0 });
	}, 240_000);

	it("a writing agent keeps its output outside the watched folder, so the rule fires once", async () => {
		writeRules([
			{
				id: "organizer",
				paths: [inbox],
				include: ["**/*.txt"],
				debounceMs: 500,
				timeoutMs: 180_000,
				agent: {
					profile: "file-organizer",
					task: `For each changed file write a one-line summary into a new file named summary.md inside ${outbox}. Do not write anywhere else.`,
				},
			},
		]);
		const { runs, errors, first } = await startLive();

		writeFileSync(join(inbox, "note.txt"), "Meeting moved to Thursday 3pm, room B.");
		const record = await first;
		// A real wait: the file watcher delivers an event of the agent's own writes some real time
		// after the run ends, and only the platform clock decides whether one arrives. Fake timers
		// cannot stand in for the OS event stream this test exercises.
		await new Promise<void>((resolve) => setTimeout(resolve, 4_000));

		expect(errors).toEqual([]);
		expect(record).toMatchObject({ exitCode: 0, timedOut: false });
		expect(existsSync(join(outbox, "summary.md"))).toBe(true);
		expect(readdirSync(inbox)).toEqual(["note.txt"]);
		expect(runs).toHaveLength(1);
	}, 300_000);
});
