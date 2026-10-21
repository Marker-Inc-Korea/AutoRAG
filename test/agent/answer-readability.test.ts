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
	ANSWER_EMPHASIS_RULE,
	ANSWER_LANGUAGE_RULE,
	ANSWER_LEAD_RULE,
	ANSWER_UNCERTAINTY_RULE,
} from "../../src/agent/answer-guidelines.ts";
import { reportSchema } from "../../src/agent/results.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

// Readability rules shape every answer, so each surface that tells the model how
// to write the answer must carry the same wording. A surface that drifts (a
// phase prompt without the language rule, a delta that demands a lead
// conclusion) makes the answer format depend on which phase produced it.

/** Rules that apply to a complete answer and to a delta against a first answer. */
const SHARED_RULES = [ANSWER_BULLET_RULE, ANSWER_EMPHASIS_RULE, ANSWER_LANGUAGE_RULE, ANSWER_UNCERTAINTY_RULE] as const;

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
			for (const rule of [ANSWER_LEAD_RULE, ...SHARED_RULES]) {
				expect(text, `${name} is missing: ${rule.slice(0, 40)}`).toContain(rule);
			}
		}
	});

	it("a delta answer keeps the shared rules but is never told to lead with a conclusion", () => {
		const delta = newAgent().buildRefinementPrompt("환불 예외는 누구 승인이 필요해?", {}, "draft [e1]", true);
		for (const rule of SHARED_RULES) expect(delta).toContain(rule);
		// The delta contract is "only corrections and new findings"; a lead conclusion would restate the first answer.
		expect(delta).not.toContain(ANSWER_LEAD_RULE);
		expect(delta).toContain("DELTA ONLY");
	});

	it("keeps the lead conclusion outside the 5-bullet cap so the cap is not silently broken", () => {
		expect(ANSWER_LEAD_RULE).toMatch(/not (counted|one of)[^.]*bullet/i);
	});

	it("puts uncertainty after the answer instead of before the conclusion", () => {
		expect(ANSWER_UNCERTAINTY_RULE).toMatch(/last line|final line|end of the answer/i);
		expect(ANSWER_UNCERTAINTY_RULE).toMatch(/never (before|ahead)/i);
	});
});
