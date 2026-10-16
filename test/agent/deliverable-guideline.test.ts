import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AssistantMessage, FauxProviderRegistration, FauxResponseStep } from "@earendil-works/pi-ai";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { reportSchema } from "../../src/agent/results.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

// Acceptance criteria for issue #1832:
// 1. Keep the 5-bullet core answer and direct-answer guidelines by default
//    unless the user explicitly requests otherwise.
// 2. When the user requests a deliverable/artifact (draft reply, letter, legal
//    opinion, summary document, standard-format case citation block, checklist),
//    produce the finished artifact as a file whose format and structure align
//    with the context and domain (rather than dumping raw prose or giving advice
//    on how to write it), while keeping the reply concise.
const REQUIRED_PHRASES = [
	"if the user asks you to draft, write, generate, or prepare a deliverable or artifact",
	"produce the finished, ready-to-use artifact as a file",
	"aligned with the context and domain",
	"instead of merely describing how to produce it",
	"Unless the user explicitly requests otherwise, keep the reply text itself within the at-most-5-bullet direct-answer format",
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
	it("states the deliverable-file rule in the system prompt while keeping the 5-bullet and direct-answer defaults", () => {
		const prompt = buildSystemPrompt({ toolNames: ["bash", "write", "edit"], modelId: "test-model" });
		expectDeliverableRule(prompt);
		expect(prompt).toContain("Unless the user explicitly requests");
		expect(prompt).toContain("5 bullet points");
		expect(prompt).toContain("Never include specific file paths");
	});

	it("states the deliverable-file rule in all three phase prompts (including delta and complete refinement)", () => {
		const agent = new AutoRAGAgent(agentOptions(fauxModel()));
		const query = "Based on the legal aid email I received from Ms. Tan, how should I reply to her professionally?";

		const search = agent.buildSearchPrompt(query, {});
		const fast = agent.buildFastAnswerPrompt(query, {}, "baseline");
		const complete = agent.buildRefinementPrompt(query, {}, undefined, false);

		for (const text of [search, fast, complete]) {
			expectDeliverableRule(text);
			expect(text).toContain("Unless the user explicitly requests");
		}

		const delta = agent.buildRefinementPrompt(
			query,
			{},
			"How to reply to Ms. Tan: 1. Acknowledge receipt [e1].",
			true,
		);
		expectDeliverableRule(delta);
		expect(delta).toContain(
			"If a deliverable or artifact was requested and the first answer could not write the file yet (or only gave advice about it), create the complete context-aligned deliverable file now during verification and report the delta concisely.",
		);
	});

	it("states the deliverable-file rule in the report answer schema", () => {
		expectDeliverableRule(answerDescription(reportSchema));
	});

	it("writes a context-aligned deliverable file via the write tool during verification and returns a concise bullet-point answer with citations", async () => {
		const seenPrompts: string[] = [];
		const emailPath = join(docs, "tan-email.txt");
		const deliverablePath = join(root, "reply-to-ms-tan.eml");
		const draftEmailContent = [
			"To: Sarah Tan <sarah.tan@example.com>",
			"Subject: Re: Legal Aid Application – Tan Mei Ling (S5634567B) | Theft (s 379 Penal Code)",
			"",
			"Dear Ms. Sarah Tan,",
			"",
			"Thank you for your email regarding Ms. Tan Mei Ling's legal aid application (S5634567B) for the s 379 Penal Code charge.",
			"We have reviewed the file and will represent her at the upcoming mention date.",
			"",
			"Yours sincerely,",
			"Legal Aid Counsel",
		].join("\n");

		const captureStep =
			(step: AssistantMessage): FauxResponseStep =>
			(context) => {
				const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
				const text =
					lastUser === undefined
						? ""
						: typeof lastUser.content === "string"
							? lastUser.content
							: lastUser.content.map((part) => (part.type === "text" ? part.text : "")).join("");
				seenPrompts.push(text);
				return step;
			};

		const model = fauxModel(
			captureStep(
				fauxAssistantMessage(
					"- Confirm conduct of Ms. Tan Mei Ling's matter before the mention date.\n- Preparing the ready-to-send reply file in verification.",
					{ stopReason: "stop" },
				),
			),
			captureStep(
				fauxAssistantMessage(
					[
						{ type: "text", text: "Writing the context-aligned .eml reply draft." },
						fauxToolCall("write", { path: deliverablePath, content: draftEmailContent }),
					],
					{ stopReason: "toolUse" },
				),
			),
			fauxAssistantMessage(
				[
					`- Prepared the ready-to-send professional reply email covering Ms. Tan Mei Ling's legal aid application (S5634567B) for the s 379 Penal Code charge [file:${emailPath}].`,
					`- Confirms representation ahead of the upcoming mention date and outlines next steps [file:${emailPath}].`,
				].join("\n"),
				{ stopReason: "stop" },
			),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments(
			"Based on the legal aid email I received from Ms. Tan, how should I reply to her professionally?",
		);

		expect(seenPrompts).toHaveLength(2);
		for (const prompt of seenPrompts) expectDeliverableRule(prompt);
		expect(existsSync(deliverablePath)).toBe(true);
		expect(readFileSync(deliverablePath, "utf8")).toContain("Dear Ms. Sarah Tan,");
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
