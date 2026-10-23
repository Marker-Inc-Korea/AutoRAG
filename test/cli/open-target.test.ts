import type { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import { openCitationTarget } from "../../src/cli/open-target.ts";

type Outcome = { readonly error: Error } | { readonly exit: number };

/** A spawn stand-in: records each launch and ends it the way `outcomes[command]` says. */
function fakeSpawn(outcomes: Record<string, Outcome>) {
	const calls: { command: string; args: string[] }[] = [];
	let pending = 0;
	const spawnFn = ((command: string, args: string[]) => {
		calls.push({ command, args });
		const child = Object.assign(new EventEmitter(), { unref: () => undefined });
		const outcome = outcomes[command] ?? { exit: 0 };
		pending++;
		queueMicrotask(() => {
			if ("error" in outcome) child.emit("error", outcome.error);
			else child.emit("close", outcome.exit);
			pending--;
		});
		return child;
	}) as unknown as typeof spawn;
	/** Yields to the microtask queue until every launch, including ones a fallback started, has ended. */
	const settled = async (): Promise<void> => {
		while (pending > 0) await Promise.resolve();
	};
	return { calls, spawnFn, settled };
}

describe("openCitationTarget", () => {
	const platformOpen = process.platform === "darwin" ? "open" : process.platform === "win32" ? "rundll32" : "xdg-open";

	it("opens a local file in Dazzi by its decoded path", async () => {
		const { calls, spawnFn, settled } = fakeSpawn({});
		openCitationTarget("file:///data/reports/q3%20(final).pdf", spawnFn);
		await settled();
		expect(calls).toEqual([{ command: "dazzi", args: ["--", "/data/reports/q3 (final).pdf"] }]);
	});

	it("falls back to the platform opener when Dazzi is not installed", async () => {
		const { calls, spawnFn, settled } = fakeSpawn({ dazzi: { error: new Error("spawn dazzi ENOENT") } });
		openCitationTarget("file:///data/a.pdf", spawnFn);
		await settled();
		expect(calls.map((call) => call.command)).toEqual(["dazzi", platformOpen]);
		expect(calls[1]?.args.at(-1)).toBe("file:///data/a.pdf");
	});

	it("falls back when Dazzi starts but exits non-zero", async () => {
		const { calls, spawnFn, settled } = fakeSpawn({ dazzi: { exit: 1 } });
		openCitationTarget("file:///data/a.pdf", spawnFn);
		await settled();
		expect(calls.map((call) => call.command)).toEqual(["dazzi", platformOpen]);
	});

	it("sends web links straight to the platform opener, never to Dazzi", async () => {
		const { calls, spawnFn, settled } = fakeSpawn({});
		openCitationTarget("https://example.com/a?b=1", spawnFn);
		await settled();
		expect(calls).toEqual([{ command: platformOpen, args: ["https://example.com/a?b=1"] }]);
	});
});
