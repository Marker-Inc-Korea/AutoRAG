import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FauxProviderRegistration } from "@earendil-works/pi-ai";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	ANSWER_BULLET_RULE,
	ANSWER_CITATION_RULE,
	ANSWER_DELIVERABLE_RULE,
	ANSWER_EMPHASIS_RULE,
	ANSWER_IMAGE_EMBED_RULE,
	ANSWER_LANGUAGE_RULE,
	ANSWER_LEAD_RULE,
	ANSWER_RELEVANCE_RULE,
	ANSWER_UNCERTAINTY_RULE,
	ANSWER_UNCONFIRMED_FAST_RULE,
	ANSWER_UNCONFIRMED_FINAL_RULE,
} from "../../src/agent/answer-guidelines.ts";
import { reportSchema } from "../../src/agent/results.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

// Readability rules shape every complete answer, so each surface that tells the
// model how to write one carries the same wording. A delta answer is the
// exception: it holds only corrections and new findings, so it carries the
// delta-specific rules plus the answer language and leans on the system prompt
// for everything else. Bullet, emphasis, lead, and uncertainty rules would pull
// it back toward re-listing the first answer.

/** Rules every complete answer carries. */
const COMPLETE_RULES = [
	ANSWER_LEAD_RULE,
	ANSWER_BULLET_RULE,
	ANSWER_EMPHASIS_RULE,
	ANSWER_LANGUAGE_RULE,
	ANSWER_UNCERTAINTY_RULE,
] as const;

/** Complete-answer rules a delta must NOT carry: they invite re-listing the first answer. */
const COMPLETE_ONLY_RULES = [
	ANSWER_LEAD_RULE,
	ANSWER_BULLET_RULE,
	ANSWER_EMPHASIS_RULE,
	ANSWER_UNCERTAINTY_RULE,
] as const;

interface TypeBoxAnswerSchema {
	properties: { answer: { description?: string } };
}

function answerDescription(schema: unknown): string {
	const description = (schema as TypeBoxAnswerSchema).properties.answer.description;
	if (description === undefined) throw new Error("answer description missing from tool schema");
	return description;
}

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-readability-"));
	mkdirSync(join(root, "docs"), { recursive: true });
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function newAgent(): AutoRAGAgent {
	const registration = registerFauxProvider({
		api: `faux-readability-${randomUUID()}`,
		models: [{ id: "faux-model", reasoning: true }],
	});
	registration.setResponses([fauxAssistantMessage("noop", { stopReason: "stop" })]);
	registrations.push(registration);
	return new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths: [join(root, "docs")],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
	});
}

describe("answer readability rules", () => {
	it("every complete-answer surface carries all five rules", () => {
		const agent = newAgent();
		const query = "환불 예외는 누구 승인이 필요해?";
		const surfaces: Record<string, string> = {
			system: buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" }),
			search: agent.buildSearchPrompt(query, {}),
			fast: agent.buildFastAnswerPrompt(query, {}, "baseline"),
			complete: agent.buildRefinementPrompt(query, {}, undefined, false),
			completeAfterDraft: agent.buildRefinementPrompt(query, {}, "draft [e1]", false),
			report: answerDescription(reportSchema),
		};
		for (const [name, text] of Object.entries(surfaces)) {
			for (const rule of COMPLETE_RULES) {
				expect(text, `${name} is missing: ${rule.slice(0, 40)}`).toContain(rule);
			}
		}
	});

	it("a delta answer keeps the answer language but none of the rules that invite re-listing", () => {
		const delta = newAgent().buildRefinementPrompt("환불 예외는 누구 승인이 필요해?", {}, "draft [e1]", true);
		expect(delta).toContain(ANSWER_LANGUAGE_RULE);
		for (const rule of COMPLETE_ONLY_RULES) expect(delta, rule.slice(0, 40)).not.toContain(rule);
		expect(delta).toContain("DELTA ONLY");
	});

	it("tells a delta answer to keep file paths and retrieval mechanics out, like every other phase", () => {
		const agent = newAgent();
		const query = "환불 예외는 누구 승인이 필요해?";
		const phrase = "Do not include specific file paths";
		for (const text of [
			agent.buildSearchPrompt(query, {}),
			agent.buildFastAnswerPrompt(query, {}, "baseline"),
			agent.buildRefinementPrompt(query, {}, undefined, false),
			agent.buildRefinementPrompt(query, {}, "draft [e1]", true),
		]) {
			expect(text).toContain(phrase);
		}
	});

	it("does not repeat the long system-prompt rules inside a delta prompt", () => {
		const agent = newAgent();
		const delta = agent.buildRefinementPrompt("환불 예외는 누구 승인이 필요해?", {}, "draft [e1]", true);
		const complete = agent.buildRefinementPrompt("환불 예외는 누구 승인이 필요해?", {}, "draft [e1]", false);
		// Deliverables, citations and image embedding are already in the system prompt; the delta carries only its own one-line additions.
		expect(delta).not.toContain(ANSWER_DELIVERABLE_RULE);
		expect(delta).not.toContain(ANSWER_CITATION_RULE);
		expect(delta).not.toContain(ANSWER_IMAGE_EMBED_RULE);
		expect(complete.length - delta.length).toBeGreaterThan(
			ANSWER_DELIVERABLE_RULE.length + ANSWER_CITATION_RULE.length + ANSWER_IMAGE_EMBED_RULE.length,
		);
	});

	it("keeps the lead conclusion outside the 5-bullet cap so the cap is not silently broken", () => {
		expect(ANSWER_LEAD_RULE).toMatch(/not (counted|one of)[^.]*bullet/i);
	});

	it("puts uncertainty after the answer instead of before the conclusion", () => {
		expect(ANSWER_UNCERTAINTY_RULE).toMatch(/last line|final line|end of the answer/i);
		expect(ANSWER_UNCERTAINTY_RULE).toMatch(/never (before|ahead)/i);
	});
});

describe("unconfirmed-answer rules", () => {
	const query = "B사와의 유지보수 계약 해지 위약금은 얼마야?";

	it("every answer surface tells the model to answer only what was asked", () => {
		const agent = newAgent();
		const surfaces: Record<string, string> = {
			system: buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" }),
			search: agent.buildSearchPrompt(query, {}),
			fast: agent.buildFastAnswerPrompt(query, {}, "baseline"),
			complete: agent.buildRefinementPrompt(query, {}, undefined, false),
			report: answerDescription(reportSchema),
		};
		for (const [name, text] of Object.entries(surfaces)) {
			expect(text, `${name} is missing the relevance rule`).toContain(ANSWER_RELEVANCE_RULE);
		}
	});

	it("a delta answer leaves out adjacent findings in one line, leaning on the system prompt for the rest", () => {
		const delta = newAgent().buildRefinementPrompt(query, {}, "draft [e1]", true);
		expect(delta).toMatch(/Leave out unrelated or merely adjacent findings/);
		expect(delta).not.toContain(ANSWER_RELEVANCE_RULE);
	});

	it("the fast answer parks an unsupported question as one line while verification keeps searching", () => {
		const agent = newAgent();
		expect(agent.buildFastAnswerPrompt(query, {}, "baseline")).toContain(ANSWER_UNCONFIRMED_FAST_RULE);
		expect(buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" })).toContain(ANSWER_UNCONFIRMED_FAST_RULE);
		expect(ANSWER_UNCONFIRMED_FAST_RULE).toMatch(/one (short )?line/i);
		expect(ANSWER_UNCONFIRMED_FAST_RULE).toMatch(/searching further/i);
	});

	it("every phase that can end the run searches further before giving up, then answers in one line", () => {
		const agent = newAgent();
		const surfaces: Record<string, string> = {
			system: buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" }),
			search: agent.buildSearchPrompt(query, {}),
			complete: agent.buildRefinementPrompt(query, {}, undefined, false),
			completeAfterDraft: agent.buildRefinementPrompt(query, {}, "draft [e1]", false),
			report: answerDescription(reportSchema),
		};
		for (const [name, text] of Object.entries(surfaces)) {
			expect(text, `${name} is missing the unconfirmed rule`).toContain(ANSWER_UNCONFIRMED_FINAL_RULE);
		}
		expect(ANSWER_UNCONFIRMED_FINAL_RULE).toMatch(/keep searching/i);
		expect(ANSWER_UNCONFIRMED_FINAL_RULE).toMatch(/one (short )?line/i);
	});

	it("a delta answer carries neither unconfirmed variant, but covers a still-unconfirmed first answer in one line", () => {
		const delta = newAgent().buildRefinementPrompt(query, {}, "draft [e1]", true);
		expect(delta).not.toContain(ANSWER_UNCONFIRMED_FAST_RULE);
		expect(delta).not.toContain(ANSWER_UNCONFIRMED_FINAL_RULE);
		expect(delta).toMatch(/still (could not be|cannot be) confirmed/i);
	});

	it("no surface invites listing clues, leads, or negative findings in place of an answer", () => {
		const agent = newAgent();
		const surfaces = [
			buildSystemPrompt({ toolNames: ["bash"], modelId: "test-model" }),
			agent.buildSearchPrompt(query, {}),
			agent.buildFastAnswerPrompt(query, {}, "baseline"),
			agent.buildRefinementPrompt(query, {}, undefined, false),
			agent.buildRefinementPrompt(query, {}, "draft [e1]", true),
			answerDescription(reportSchema),
		];
		for (const text of surfaces) {
			expect(text).not.toMatch(/mention (those |them |the )?clues|partial leads exist|report a negative result/i);
		}
	});
});
