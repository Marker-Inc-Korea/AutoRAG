import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessage, FauxProviderRegistration, FauxResponseStep } from "@earendil-works/pi-ai";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { reportSchema } from "../../src/agent/results.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

// Acceptance criteria for issue #1832: every prompt/tool that shapes `answer`
// must instruct the model to put a requested deliverable/artifact (draft reply,
// letter, legal opinion, summary document, standard-format case citation,
// checklist) directly in the answer — ready to use — instead of describing how
// to produce it, relaxing the 5-bullet cap and "direct answer only" wording
// while still keeping source paths and retrieval mechanics out.
const REQUIRED_PHRASES = [
	"if the user asks you to draft, write, generate, or prepare something",
	"put the finished artifact itself in the answer",
	"instead of describing how to produce it",
	'The 5-bullet cap and the "direct answer only" wording do not apply to a requested artifact',
	"keep paths and retrieval mechanics out, but do not compress away the artifact",
] as const;

function expectDeliverableRule(text: string): void {
	for (const phrase of REQUIRED_PHRASES) expect(text).toContain(phrase);
}

/** Runtime TypeBox schemas expose `description`, which their TS type omits. */
interface TypeBoxAnswerSchema {
	properties: { answer: { description?: string } };
}

function answerDescription(schema: unknown): string {
	const typed = schema as TypeBoxAnswerSchema;
	const description = typed.properties.answer.description;
	if (description === undefined) throw new Error("answer description missing from tool schema");
	return description;
}

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-deliverable-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(
		join(docs, "tan-email.txt"),
		[
			"From: Sarah Tan <sarah.tan@example.com>",
			"Subject: Legal Aid Application – Tan Mei Ling (S5634567B) | Theft (s 379 Penal Code)",
			"Please advise on the status of Ms. Tan Mei Ling's application and next steps before the mention date.",
		].join("\n"),
	);
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...steps: FauxResponseStep[]) {
	const registration = registerFauxProvider({
		api: `faux-deliverable-${randomUUID()}`,
		models: [{ id: "faux-model", reasoning: true }],
	});
	registration.setResponses(steps.length > 0 ? steps : [fauxAssistantMessage("noop", { stopReason: "stop" })]);
	registrations.push(registration);
	return registration.getModel();
}

function agentOptions(model: AutoRAGAgentOptions["model"]): AutoRAGAgentOptions {
	return {
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
	};
}

describe("answer deliverable/artifact guideline (#1832)", () => {
	it("states the deliverable rule in the system prompt while keeping the 5-bullet default and no-path rule", () => {
		const prompt = buildSystemPrompt({ toolNames: [], modelId: "test-model" });
		expectDeliverableRule(prompt);
		expect(prompt).toContain("5 bullet points");
		expect(prompt).toContain("Never include specific file paths");
	});

	it("states the deliverable rule in all three phase prompts (including delta and complete refinement)", () => {
		const agent = new AutoRAGAgent(agentOptions(fauxModel()));
		const query = "Based on the legal aid email I received from Ms. Tan, how should I reply to her professionally?";

		const search = agent.buildSearchPrompt(query, {});
		const fast = agent.buildFastAnswerPrompt(query, {}, "baseline");
		const complete = agent.buildRefinementPrompt(query, {}, undefined, false);

		for (const text of [search, fast, complete]) expectDeliverableRule(text);

		const delta = agent.buildRefinementPrompt(
			query,
			{},
			"How to reply to Ms. Tan: 1. Acknowledge receipt [e1].",
			true,
		);
		expectDeliverableRule(delta);
		expect(delta).toContain(
			"If the first answer omitted a requested artifact or only described how to write it, output the complete ready-to-use artifact now.",
		);
	});

	it("states the deliverable rule in the report answer schema", () => {
		expectDeliverableRule(answerDescription(reportSchema));
	});

	it("delivers the deliverable rule to the model during searchDocuments and preserves a multi-paragraph draft reply with inline citations", async () => {
		const seenPrompts: string[] = [];
		const captureStep =
			(reply: string): FauxResponseStep =>
			(context) => {
				const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
				const text =
					lastUser === undefined
						? ""
						: typeof lastUser.content === "string"
							? lastUser.content
							: lastUser.content.map((part) => (part.type === "text" ? part.text : "")).join("");
				seenPrompts.push(text);
				return fauxAssistantMessage(reply, { stopReason: "stop" }) as AssistantMessage;
			};

		const emailPath = join(docs, "tan-email.txt");
		const draftReply = [
			"Subject: Re: Legal Aid Application – Tan Mei Ling (S5634567B) | Theft (s 379 Penal Code)",
			"",
			"Dear Ms. Sarah Tan,",
			"",
			`Thank you for your email regarding Ms. Tan Mei Ling's legal aid application (S5634567B) for the s 379 Penal Code charge [file:${emailPath}].`,
			"",
			"We have reviewed the file and will represent her at the upcoming mention date.",
			"",
			"Yours sincerely,",
			"Legal Aid Counsel",
		].join("\n");

		const model = fauxModel(captureStep("Drafting reply..."), captureStep(draftReply));
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments(
			"Based on the legal aid email I received from Ms. Tan, how should I reply to her professionally?",
		);

		expect(seenPrompts).toHaveLength(2);
		for (const prompt of seenPrompts) expectDeliverableRule(prompt);
		expect(response.answer).toContain("Subject: Re: Legal Aid Application – Tan Mei Ling (S5634567B)");
		expect(response.answer).toContain("Dear Ms. Sarah Tan,");
		expect(response.answer).toContain("Yours sincerely,");
		expect(response.answer).toContain("[1]");
		expect(response.results[0]?.source).toBe(emailPath);
	});

	it("preserves neutral citations and pinpoint paragraph numbers inside a case-citation deliverable (#1832)", async () => {
		const precedentPath = join(docs, "tan-email.txt");
		const citationBlock = [
			`*WXY v ZAB* [2024] SGHCF 24 at [18]–[22] [file:${precedentPath}].`,
			"",
			"**[18]** In assessing division of matrimonial assets under s 112 of the Women's Charter, the structured approach in *ANJ v ANK* [2015] 4 SLR 1043 applies.",
			"",
			"At [22], the court held that an applicant seeking interim maintenance must show immediate financial need.",
		].join("\n");

		const model = fauxModel(
			fauxAssistantMessage("Preparing citation block...", { stopReason: "stop" }),
			fauxAssistantMessage(citationBlock, { stopReason: "stop" }),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments("Generate a standard-format case citation for direct insertion.");

		expect(response.answer).toContain("[2024] SGHCF 24 at [18]–[22] [1].");
		expect(response.answer).toContain("**[18]** In assessing division");
		expect(response.answer).toContain("At [22], the court held");
		expect(response.results[0]?.source).toBe(precedentPath);
	});
});
