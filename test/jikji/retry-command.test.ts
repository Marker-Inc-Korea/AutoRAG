import { describe, expect, it } from "vitest";
import { parseJikjiRetryCommand } from "../../src/jikji/retry-command.ts";

const PROOF = "ab9ebb1eb3f6e5460b4f1c23";

describe("parseJikjiRetryCommand", () => {
	it("reads root, rewritten query, top-k and proof from the command jikji 0.2.1 emits", () => {
		const step = parseJikjiRetryCommand(
			`jikji discover '/tmp/jk.fcvs/root' 'unrelatedterm zzzqqq' --top-k 20 --after-jikji-retry --retry-proof '${PROOF}' --json`,
			PROOF,
		);
		expect(step).toEqual({ root: "/tmp/jk.fcvs/root", query: "unrelatedterm zzzqqq", topK: 20, proof: PROOF });
	});

	it("takes the proof from the command when the envelope carries none", () => {
		const step = parseJikjiRetryCommand(
			`jikji discover '/r' 'q' --after-jikji-retry --retry-proof '${PROOF}' --json`,
			undefined,
		);
		expect(step).toMatchObject({ proof: PROOF });
		expect(step?.topK).toBeUndefined();
	});

	it("round-trips a query containing a single quote, spaces and unicode", () => {
		const step = parseJikjiRetryCommand(
			`jikji discover '/tmp/my docs' 'it'\\''s 환불 정책' --top-k 20 --after-jikji-retry --retry-proof '${PROOF}' --json`,
			PROOF,
		);
		expect(step).toMatchObject({ root: "/tmp/my docs", query: "it's 환불 정책" });
	});

	it("keeps a query that starts with dashes as a positional", () => {
		const step = parseJikjiRetryCommand(
			`jikji discover '/r' '--first draft' --after-jikji-retry --retry-proof '${PROOF}' --json`,
			PROOF,
		);
		expect(step?.query).toBe("--first draft");
	});

	it("accepts an absolute jikji binary path", () => {
		const step = parseJikjiRetryCommand(
			`/Users/me/.cargo/bin/jikji discover '/r' 'q' --after-jikji-retry --retry-proof '${PROOF}' --json`,
			PROOF,
		);
		expect(step).toBeDefined();
	});

	it.each([
		["a subcommand other than discover", `jikji find '/r' 'q' --after-jikji-retry --retry-proof '${PROOF}' --json`],
		["no retry proof", "jikji discover '/r' 'q' --after-jikji-retry --json"],
		["no --after-jikji-retry", `jikji discover '/r' 'q' --retry-proof '${PROOF}' --json`],
		[
			"a flag it does not know",
			`jikji discover '/r' 'q' --after-jikji-retry --retry-proof '${PROOF}' --rm-rf --json`,
		],
		["a non-jikji program", `rm discover '/r' 'q' --after-jikji-retry --retry-proof '${PROOF}' --json`],
		[
			"shell chaining outside quotes",
			`jikji discover '/r' 'q' --after-jikji-retry --retry-proof '${PROOF}' ; rm -rf /`,
		],
		[
			"command substitution outside quotes",
			`jikji discover '/r' $(whoami) --after-jikji-retry --retry-proof '${PROOF}'`,
		],
		["an unterminated quote", `jikji discover '/r 'q --after-jikji-retry --retry-proof '${PROOF}`],
		[
			"a non-numeric top-k",
			`jikji discover '/r' 'q' --top-k many --after-jikji-retry --retry-proof '${PROOF}' --json`,
		],
		["a missing positional", `jikji discover '/r' --after-jikji-retry --retry-proof '${PROOF}' --json`],
	])("refuses %s", (_label, command) => {
		expect(parseJikjiRetryCommand(command, PROOF)).toBeUndefined();
	});

	it("refuses a command whose proof differs from the envelope's retry_proof", () => {
		const step = parseJikjiRetryCommand(
			`jikji discover '/r' 'q' --after-jikji-retry --retry-proof 'other' --json`,
			PROOF,
		);
		expect(step).toBeUndefined();
	});
});
