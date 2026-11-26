import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FauxProviderRegistration, FauxResponseStep } from "@earendil-works/pi-ai";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

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
		jev: baselineJev(),
	};
}

describe("answer deliverable/artifact behavior (#1832)", () => {
	it("writes a context-aligned deliverable file via the write tool during verification and returns the created file location and concise summary without dumping the full text", async () => {
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

		const model = fauxModel(
			fauxAssistantMessage(
				"- Confirm conduct of Ms. Tan Mei Ling's matter before the mention date.\n- Preparing the ready-to-send reply file in verification.",
				{ stopReason: "stop" },
			),
			fauxAssistantMessage(
				[
					{ type: "text", text: "Writing the context-aligned .eml reply draft." },
					fauxToolCall("write", { path: deliverablePath, content: draftEmailContent }),
				],
				{ stopReason: "toolUse" },
			),
			fauxAssistantMessage(
				[
					`- Created the ready-to-send reply email file at \`${deliverablePath}\` covering Ms. Tan Mei Ling's legal aid application (S5634567B) for the s 379 Penal Code charge [file:${emailPath}].`,
					`- Summarizes confirmation of legal representation ahead of the upcoming mention date and outlines next steps [file:${emailPath}].`,
				].join("\n"),
				{ stopReason: "stop" },
			),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments(
			"Based on the legal aid email I received from Ms. Tan, how should I reply to her professionally?",
		);

		expect(existsSync(deliverablePath)).toBe(true);
		expect(readFileSync(deliverablePath, "utf8")).toContain("Dear Ms. Sarah Tan,");
		expect(response.answer).toContain(deliverablePath);
		expect(response.answer).not.toContain("Dear Ms. Sarah Tan,");
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
