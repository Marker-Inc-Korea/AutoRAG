import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { main } from "../../src/cli/index.ts";

let root: string;
let home: string;
let watched: string;
let prevHome: string | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-automate-"));
	home = join(root, "autorag-home");
	watched = join(root, "watched");
	mkdirSync(home, { recursive: true });
	mkdirSync(watched, { recursive: true });
	prevHome = process.env.AUTORAG_HOME;
	process.env.AUTORAG_HOME = home;
});

afterEach(() => {
	if (prevHome === undefined) delete process.env.AUTORAG_HOME;
	else process.env.AUTORAG_HOME = prevHome;
	rmSync(root, { recursive: true, force: true });
});

async function runCli(argv: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
	const out: string[] = [];
	const err: string[] = [];
	const origOut = process.stdout.write.bind(process.stdout);
	const origErr = process.stderr.write.bind(process.stderr);
	process.stdout.write = ((chunk: string | Uint8Array) => {
		out.push(String(chunk));
		return true;
	}) as typeof process.stdout.write;
	process.stderr.write = ((chunk: string | Uint8Array) => {
		err.push(String(chunk));
		return true;
	}) as typeof process.stderr.write;
	try {
		const code = await main(argv);
		return { code, stdout: out.join(""), stderr: err.join("") };
	} finally {
		process.stdout.write = origOut;
		process.stderr.write = origErr;
	}
}

describe("autorag automate run", () => {
	it("runs the matching automation with repeatable --event flags and forwards exit status", async () => {
		const script = join(root, "dump.mjs");
		writeFileSync(script, "let s='';for await(const c of process.stdin)s+=c;console.log(s.trim());");
		writeFileSync(
			join(home, "automations.json"),
			JSON.stringify({
				automations: [{ id: "pdf", paths: [watched], include: ["**/*.pdf"], run: ["node", script] }],
			}),
		);
		const a = join(watched, "a.pdf");
		const b = join(watched, "sub", "b.pdf");
		const ignored = join(watched, "c.txt");

		const res = await runCli([
			"automate",
			"run",
			"pdf",
			"--event",
			`create:${a}`,
			"--event",
			`update:${b}`,
			"--event",
			`create:${ignored}`,
			"--json",
		]);
		expect(res.code).toBe(0);
		const payload = JSON.parse(res.stdout);
		expect(payload.ok).toBe(true);
		expect(payload.matchedEvents).toBe(2);
		expect(payload.ignoredEvents).toBe(1);
		expect(JSON.parse(payload.stdout)).toEqual([
			{ type: "create", path: a },
			{ type: "update", path: b },
		]);
	});

	it("returns exit 1 when the command exits non-zero", async () => {
		const script = join(root, "fail.mjs");
		writeFileSync(script, "console.error('boom');process.exit(7);");
		writeFileSync(
			join(home, "automations.json"),
			JSON.stringify({ automations: [{ id: "f", paths: [watched], run: ["node", script] }] }),
		);
		const res = await runCli(["automate", "run", "f", "--json"]);
		expect(res.code).toBe(1);
		expect(JSON.parse(res.stdout).exitCode).toBe(7);
	});

	it("returns exit 2 for an unknown id, bad --event, or when every --event was filtered out", async () => {
		writeFileSync(
			join(home, "automations.json"),
			JSON.stringify({
				automations: [{ id: "pdf", paths: [watched], include: ["**/*.pdf"], run: ["node", "-e", "0"] }],
			}),
		);
		expect((await runCli(["automate", "run", "nope"])).code).toBe(2);
		expect((await runCli(["automate", "run", "pdf", "--event", "bogus:/a"])).code).toBe(2);
		expect((await runCli(["automate", "run", "pdf", "--event", `create:${join(watched, "a.txt")}`])).code).toBe(2);
	});

	it("status reports service state and the last run record for each automation", async () => {
		writeFileSync(
			join(home, "automations.json"),
			JSON.stringify({ automations: [{ id: "pdf", paths: [watched], run: ["node", "-e", "0"] }] }),
		);
		await runCli(["automate", "run", "pdf"]);
		const status = await runCli(["automate", "status", "--json"]);
		expect(status.code).toBe(0);
		const body = JSON.parse(status.stdout);
		expect(body.ok).toBe(true);
		expect(body.automations[0].id).toBe("pdf");
		expect(body.automations[0].lastRun.exitCode).toBe(0);
	});
});
