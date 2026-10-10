import { Value } from "typebox/value";
import { describe, expect, it } from "vitest";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import {
	type AutoRAGFastAnswerDetails,
	createEmitFastAnswerTool,
	EMIT_FAST_ANSWER_TOOL_NAME,
} from "../../src/agent/fast-answer-tool.ts";

function ledgerWith() {
	const ledger = new EvidenceLedger();
	const id = ledger.registerResult("baseline", {
		id: "minsync:doc#1",
		source: "/docs/refund.txt",
		content: "Refund exceptions require director approval.",
		score: 1,
		metadata: { method: "minsync" },
	});
	return { ledger, id };
}

const unit = { number: 1, title: "Refund rule", summary: "Director approval", confidence: 0.6 };

describe("createEmitFastAnswerTool", () => {
	it("takes evidence ids per result and has no model-written sources list", () => {
		const tool = createEmitFastAnswerTool(() => {}, { ledger: new EvidenceLedger() });
		expect(tool.name).toBe(EMIT_FAST_ANSWER_TOOL_NAME);
		expect(Object.keys(tool.parameters.properties ?? {}).sort()).toEqual(["answer", "results"]);
		expect(Value.Check(tool.parameters, { answer: "a [1]", results: [{ ...unit, refs: ["e1"] }] })).toBe(true);
	});

	it("derives sources from the cited evidence so the model cannot omit or misquote them", async () => {
		const { ledger, id } = ledgerWith();
		let captured: AutoRAGFastAnswerDetails | undefined;
		const tool = createEmitFastAnswerTool(
			(details) => {
				captured = details;
			},
			{ ledger },
		);

		const result = await tool.execute("c1", {
			answer: "Director approval is required [1]",
			results: [{ ...unit, refs: [id] }],
		});

		expect(result.details.sources).toEqual([{ number: 1, source: "/docs/refund.txt" }]);
		expect(captured).toBe(result.details);
		expect(result.terminate).toBeUndefined();
	});

	it("carries the harness-recorded evidence per result, not the model's excerpt", async () => {
		const { ledger, id } = ledgerWith();
		const tool = createEmitFastAnswerTool(() => {}, { ledger });

		const result = await tool.execute("c1b", {
			answer: "Director approval is required [1]",
			results: [{ ...unit, evidence: [{ excerpt: "Refund ... approval" }], refs: [id] }],
		});

		expect(result.details.evidenceRefs).toEqual([
			{
				number: 1,
				refs: [
					expect.objectContaining({
						method: "minsync",
						source: "/docs/refund.txt",
						content: "Refund exceptions require director approval.",
						retrievalResultId: "minsync:doc#1",
					}),
				],
			},
		]);
	});

	it("keeps an answer with no results valid for general-knowledge replies", async () => {
		const tool = createEmitFastAnswerTool(() => {}, { ledger: new EvidenceLedger() });
		const result = await tool.execute("c2", { answer: "Hello!", results: [] });
		expect(result.details).toMatchObject({ answer: "Hello!", results: [], sources: [] });
	});

	it("lets a result with no refs through (fast phase never blocks on a source)", async () => {
		const tool = createEmitFastAnswerTool(() => {}, { ledger: new EvidenceLedger() });
		const result = await tool.execute("c3", { answer: "partial [1]", results: [unit] });
		expect(result.details.sources).toEqual([]);
		expect(result.details.results[0]?.title).toBe("Refund rule");
	});

	it("rejects an invented evidence id so the model re-emits", async () => {
		const tool = createEmitFastAnswerTool(() => {}, { ledger: new EvidenceLedger() });
		await expect(tool.execute("c4", { answer: "x [1]", results: [{ ...unit, refs: ["e42"] }] })).rejects.toThrow(
			/e42/u,
		);
	});

	it("rejects citations that point at no result", async () => {
		const tool = createEmitFastAnswerTool(() => {}, { ledger: new EvidenceLedger() });
		await expect(tool.execute("c5", { answer: "x [3]", results: [unit] })).rejects.toThrow(/\[3\]/u);
	});
});
