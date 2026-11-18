import { describe, expect, it } from "vitest";
import {
	AnswerRevisionSession,
	type Revision,
	type RevisionCitations,
	type RevisionOutcome,
	splitClaims,
} from "../../src/agent/answer-revision.ts";

/** Resolves `[e1]`..`[e9]` and `[file:/ok]`; everything else, user-facing `[n]` numbers included, is unresolved. */
const citations: RevisionCitations = {
	inspect(text) {
		const unresolved: string[] = [];
		let resolved = 0;
		for (const match of text.matchAll(/\[(e\d+|file:[^\]]*|\d+)\]/gu)) {
			const id = match[1] ?? "";
			if (/^e[1-9]$/u.test(id) || id === "file:/ok") resolved += 1;
			else unresolved.push(match[0]);
		}
		return { resolved, unresolved };
	},
	// The user-facing form: `[e1]` -> `[1]`.
	display: (text) => text.replace(/\[e(\d)\]/gu, "[$1]"),
};

function accepted(outcome: RevisionOutcome): Revision {
	if (!outcome.ok) throw new Error(`expected the revision to be accepted, got: ${outcome.message}`);
	return outcome.revision;
}

function rejected(outcome: RevisionOutcome): string {
	if (outcome.ok) throw new Error("expected the revision to be rejected");
	return outcome.message;
}

function reassembled(text: string): string {
	return splitClaims(text)
		.map((claim) => claim.text + claim.sep)
		.join("");
}

describe("splitClaims", () => {
	it("gives every bullet its own claim and numbers them c1..cN in order", () => {
		const claims = splitClaims("- Refunds need approval [e1]\n- The cap is 50만 원 [e2]\n- Paid monthly [e3]");
		expect(claims.map((claim) => claim.id)).toEqual(["c1", "c2", "c3"]);
		expect(claims.map((claim) => claim.text)).toEqual([
			"- Refunds need approval [e1]",
			"- The cap is 50만 원 [e2]",
			"- Paid monthly [e3]",
		]);
	});

	it("keeps a bullet's wrapped continuation lines with the bullet", () => {
		const claims = splitClaims("- Refunds need approval\n  from a director [e1]\n- Paid monthly [e2]");
		expect(claims.map((claim) => claim.text)).toEqual([
			"- Refunds need approval\n  from a director [e1]",
			"- Paid monthly [e2]",
		]);
	});

	it("splits a paragraph into sentences, in Korean and English", () => {
		const claims = splitClaims(
			"환불은 이사 승인이 필요합니다 [e1]. 한도는 50만 원입니다 [e2]. It is paid monthly. Done!",
		);
		expect(claims.map((claim) => claim.text)).toEqual([
			"환불은 이사 승인이 필요합니다 [e1].",
			"한도는 50만 원입니다 [e2].",
			"It is paid monthly.",
			"Done!",
		]);
	});

	it("splits CJK sentences that follow each other without a space", () => {
		expect(splitClaims("승인이 필요합니다。한도는 있습니다。").map((claim) => claim.text)).toEqual([
			"승인이 필요합니다。",
			"한도는 있습니다。",
		]);
	});

	it("keeps a citation that follows the full stop with the sentence it backs", () => {
		const claims = splitClaims("The cap is 50만 원. [e2] It is paid monthly. [file:/ok] Next.");
		expect(claims.map((claim) => claim.text)).toEqual([
			"The cap is 50만 원. [e2]",
			"It is paid monthly. [file:/ok]",
			"Next.",
		]);
	});

	it("keeps every citation of a run, comma lists and spaces included, with the sentence they back", () => {
		const claims = splitClaims("Paid monthly. [e1, e2]\t[e3] [file:/a b] Next. [e4 ,e5]");
		expect(claims.map((claim) => claim.text)).toEqual(["Paid monthly. [e1, e2]\t[e3] [file:/a b]", "Next. [e4 ,e5]"]);
	});

	it("leaves a malformed citation to the next sentence instead of swallowing it", () => {
		for (const broken of ["[e]", "[e1", "[e1,]", "[e1 e2]", "[file:/a", "[file:/a\nb]", "[x1]"]) {
			const claims = splitClaims(`Paid monthly. ${broken} Next.`);
			expect(claims[0]?.text).toBe("Paid monthly.");
		}
	});

	it("splits a long hostile line in linear time", () => {
		for (const hostile of [
			"[file:".repeat(60_000),
			`Done.${"\t".repeat(100_000)}x`,
			`Done. ${"[e1]".repeat(60_000)}`,
			`Done. [e1${",e1".repeat(60_000)}]`,
		]) {
			const started = performance.now();
			splitClaims(hostile);
			expect(performance.now() - started).toBeLessThan(1_000);
		}
	});

	it("does not split inside decimals, versions, or file paths", () => {
		const claims = splitClaims("Version 3.5 shipped on /a/b.txt yesterday. Next.");
		expect(claims.map((claim) => claim.text)).toEqual(["Version 3.5 shipped on /a/b.txt yesterday.", "Next."]);
	});

	it("keeps a fenced code block and a table each as one claim", () => {
		const text = ["Intro.", "```ts", "const a = 1. b;", "```", "| a | b |", "|---|---|", "| 1 | 2 |", "Outro."].join(
			"\n",
		);
		expect(splitClaims(text).map((claim) => claim.text)).toEqual([
			"Intro.",
			["```ts", "const a = 1. b;", "```"].join("\n"),
			["| a | b |", "|---|---|", "| 1 | 2 |"].join("\n"),
			"Outro.",
		]);
	});

	it("keeps an unclosed code fence as one claim to the end", () => {
		expect(splitClaims("```\nlet a = 1. b\nmore").map((claim) => claim.text)).toEqual(["```\nlet a = 1. b\nmore"]);
	});

	it("reassembles to exactly the trimmed input", () => {
		for (const text of [
			"- a [e1]\n- b [e2]",
			"Para one. Still one.\n\nPara two.\n\n- x\n- y",
			"한 문장입니다. 두 번째 문장입니다.\n\n## 제목\n본문입니다.",
			"```\ncode. here\n```\n\nAfter.",
			"line one.\r\nline two.",
			"  leading and trailing blank  \n\n",
		]) {
			expect(reassembled(text.trim())).toBe(text.trim());
		}
	});

	it("returns no claims for blank text", () => {
		expect(splitClaims("  \n ")).toEqual([]);
	});
});

describe("AnswerRevisionSession", () => {
	const fast = "- Refunds need team-lead approval [e1]\n- The cap is 50만 원 [e2]\n- Paid monthly [e3]";

	function open(): AnswerRevisionSession {
		return new AnswerRevisionSession(fast, citations);
	}

	it("also reports a revision as the model wrote it, for a surface that shows the model's own [eN] citations", () => {
		const outcome = open().apply({
			op: "replace",
			claimId: "c1",
			text: "- Refunds need director approval [e4]",
			reason: "July policy",
		});
		if (!outcome.ok) throw new Error(outcome.message);
		expect(outcome.written).toEqual({
			original: "- Refunds need team-lead approval [e1]",
			replacement: "- Refunds need director approval [e4]",
		});
		expect(outcome.revision.replacement).toBe("- Refunds need director approval [4]");
	});

	it("reports a retraction as written without a replacement, and an addition without an original", () => {
		const retracted = open().apply({ op: "retract", claimId: "c2", reason: "x" });
		if (!retracted.ok) throw new Error(retracted.message);
		expect(retracted.written).toEqual({ original: "- The cap is 50만 원 [e2]" });
		const added = open().apply({ op: "add", text: "New [e5]", reason: "x" });
		if (!added.ok) throw new Error(added.message);
		expect(added.written).toEqual({ replacement: "New [e5]" });
	});

	it("lists the claims as the model wrote them, so it can name them and its own [eN] ids stay valid", () => {
		expect(open().claims()).toEqual([
			{ id: "c1", text: "- Refunds need team-lead approval [e1]" },
			{ id: "c2", text: "- The cap is 50만 원 [e2]" },
			{ id: "c3", text: "- Paid monthly [e3]" },
		]);
	});

	it("replaces a claim, reporting what the user saw and what replaces it", () => {
		const session = open();
		const revision = accepted(
			session.apply({
				op: "replace",
				claimId: "c1",
				text: "- Refunds need director approval [e4]",
				reason: "July policy moved approval to directors",
			}),
		);
		expect(revision).toEqual({
			op: "replace",
			claimId: "c1",
			original: "- Refunds need team-lead approval [1]",
			replacement: "- Refunds need director approval [4]",
			reason: "July policy moved approval to directors",
		});
		expect(session.revisedText()).toBe(
			"- Refunds need director approval [e4]\n- The cap is 50만 원 [e2]\n- Paid monthly [e3]",
		);
	});

	it("retracts a claim and drops it, with its separator, from the revised text", () => {
		const session = open();
		const revision = accepted(session.apply({ op: "retract", claimId: "c2", reason: "no source states a cap" }));
		expect(revision).toEqual({
			op: "retract",
			claimId: "c2",
			original: "- The cap is 50만 원 [2]",
			reason: "no source states a cap",
		});
		expect(revision).not.toHaveProperty("replacement");
		expect(session.revisedText()).toBe("- Refunds need team-lead approval [e1]\n- Paid monthly [e3]");
	});

	it("drops a retracted last claim without leaving a trailing separator", () => {
		const session = open();
		session.apply({ op: "retract", claimId: "c3", reason: "stale" });
		expect(session.revisedText()).toBe("- Refunds need team-lead approval [e1]\n- The cap is 50만 원 [e2]");
	});

	it("retracts one sentence of a paragraph and keeps the rest readable", () => {
		const session = new AnswerRevisionSession("A is true [e1]. B is true [e2]. C is true [e3].", citations);
		session.apply({ op: "retract", claimId: "c2", reason: "wrong" });
		expect(session.revisedText()).toBe("A is true [e1]. C is true [e3].");
	});

	it("appends an added finding after the existing answer", () => {
		const session = open();
		const revision = accepted(
			session.apply({ op: "add", text: "- Exceptions are logged [e5]", reason: "found in the audit policy" }),
		);
		expect(revision).toEqual({
			op: "add",
			replacement: "- Exceptions are logged [5]",
			reason: "found in the audit policy",
		});
		expect(session.revisedText()).toBe(`${fast}\n- Exceptions are logged [e5]`);
	});

	it("separates a prose addition from the answer with a blank line", () => {
		const session = new AnswerRevisionSession("Refunds need approval [e1].", citations);
		session.apply({ op: "add", text: "Exceptions are logged [e5].", reason: "audit policy" });
		expect(session.revisedText()).toBe("Refunds need approval [e1].\n\nExceptions are logged [e5].");
	});

	it("shows only the addition when every claim was retracted", () => {
		const session = new AnswerRevisionSession("Only claim [e1].", citations);
		session.apply({ op: "retract", claimId: "c1", reason: "wrong" });
		session.apply({ op: "add", text: "Real answer [e2].", reason: "found it" });
		expect(session.revisedText()).toBe("Real answer [e2].");
	});

	it("confirms a claim without changing the text and counts it as reviewed", () => {
		const session = open();
		const revision = accepted(session.apply({ op: "confirm", claimId: "c3", reason: "matches the payroll policy" }));
		expect(revision).toMatchObject({ op: "confirm", claimId: "c3", original: "- Paid monthly [3]" });
		expect(session.revisedText()).toBe(fast);
		expect(session.unreviewed()).toEqual(["c1", "c2"]);
	});

	it("rejects an unknown claim id and names the valid ones", () => {
		const message = rejected(open().apply({ op: "retract", claimId: "c9", reason: "x" }));
		expect(message).toContain("c9");
		expect(message).toContain("c1, c2, c3");
	});

	it("rejects a claim id on add and a missing claim id on the other ops", () => {
		rejected(open().apply({ op: "replace", text: "x [e1]", reason: "x" }));
		rejected(open().apply({ op: "retract", reason: "x" }));
		rejected(open().apply({ op: "add", claimId: "c1", text: "x [e1]", reason: "x" }));
	});

	it("requires a reason, and text exactly where the op carries text", () => {
		rejected(open().apply({ op: "retract", claimId: "c1", reason: "   " }));
		rejected(open().apply({ op: "replace", claimId: "c1", reason: "x" }));
		rejected(open().apply({ op: "add", reason: "x" }));
		rejected(open().apply({ op: "retract", claimId: "c1", text: "stray", reason: "x" }));
		rejected(open().apply({ op: "confirm", claimId: "c1", text: "stray", reason: "x" }));
	});

	it("rejects a replacement of a cited claim that cites nothing, or cites evidence the run never retrieved", () => {
		expect(rejected(open().apply({ op: "replace", claimId: "c1", text: "Directors approve.", reason: "x" }))).toMatch(
			/cite/i,
		);
		expect(
			rejected(open().apply({ op: "replace", claimId: "c1", text: "Directors approve [e42].", reason: "x" })),
		).toContain("[e42]");
	});

	it("lets an uncited claim be replaced by an uncited one, since it never claimed a source", () => {
		const session = new AnswerRevisionSession("It takes about three days.", citations);
		const revision = accepted(
			session.apply({ op: "replace", claimId: "c1", text: "It takes about five days.", reason: "calculator" }),
		);
		expect(revision.replacement).toBe("It takes about five days.");
	});

	it("lets an added finding stand without a citation (a calculation, a created file), but never with an unresolved one", () => {
		accepted(open().apply({ op: "add", text: "The total is 12 days.", reason: "calculator" }));
		expect(rejected(open().apply({ op: "add", text: "Logged [e42].", reason: "x" }))).toContain("[e42]");
	});

	it("rejects a user-facing [n] marker, which the model must not write", () => {
		const message = rejected(open().apply({ op: "add", text: "Directors approve [2].", reason: "x" }));
		expect(message).toContain("[2]");
		expect(message).toMatch(/\[eN\]/);
	});

	it("revises a claim once: a retracted or replaced claim cannot be revised again", () => {
		const session = open();
		session.apply({ op: "retract", claimId: "c1", reason: "wrong" });
		expect(rejected(session.apply({ op: "replace", claimId: "c1", text: "x [e1]", reason: "y" }))).toContain(
			"retracted",
		);
		session.apply({ op: "replace", claimId: "c2", text: "y [e2]", reason: "z" });
		expect(rejected(session.apply({ op: "retract", claimId: "c2", reason: "again" }))).toContain("replaced");
	});

	it("lets a confirmed claim still be corrected later, but not confirmed twice", () => {
		const session = open();
		session.apply({ op: "confirm", claimId: "c1", reason: "looks right" });
		rejected(session.apply({ op: "confirm", claimId: "c1", reason: "still right" }));
		accepted(session.apply({ op: "retract", claimId: "c1", reason: "found a conflict" }));
		expect(session.revisedText()).not.toContain("Refunds need team-lead");
	});

	it("numbers a new source for the user only when a revision is applied, never when it is only checked", () => {
		const numbered: string[] = [];
		const session = new AnswerRevisionSession(fast, {
			...citations,
			display: (text) => {
				numbered.push(text);
				return citations.display(text);
			},
		});
		session.check({ op: "replace", claimId: "c1", text: "Directors approve [e4].", reason: "x" });
		session.check({ op: "add", text: "New [e5].", reason: "x" });
		expect(numbered).toEqual([]);
		session.apply({ op: "add", text: "New [e5].", reason: "x" });
		expect(numbered).toEqual(["New [e5]."]);
	});

	it("keeps the revisions in the order they were applied", () => {
		const session = open();
		session.apply({ op: "retract", claimId: "c2", reason: "a" });
		session.apply({ op: "confirm", claimId: "c3", reason: "b" });
		session.apply({ op: "add", text: "New [e5].", reason: "c" });
		expect(session.revisions().map((revision) => revision.op)).toEqual(["retract", "confirm", "add"]);
	});

	it("does not record a rejected revision", () => {
		const session = open();
		session.apply({ op: "retract", claimId: "c9", reason: "x" });
		expect(session.revisions()).toEqual([]);
	});

	it("reports whether the revised answer changed anything", () => {
		const session = open();
		expect(session.changed()).toBe(false);
		session.apply({ op: "confirm", claimId: "c1", reason: "ok" });
		expect(session.changed()).toBe(false);
		session.apply({ op: "retract", claimId: "c2", reason: "bad" });
		expect(session.changed()).toBe(true);
	});
});
