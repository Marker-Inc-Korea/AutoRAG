import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { FauxProviderRegistration } from "@earendil-works/pi-ai";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { reportSchema } from "../../src/agent/results.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";

// The acceptance criteria for issue #1790: every prompt/tool that shapes
// `answer` must allow exactly one file-path exception — a markdown image embed
// whose path is a real result source — while the no-file-paths rule otherwise
// stands.
const REQUIRED_PHRASES = [
	"when a retrieved result is itself an image file (png, jpg, jpeg, gif, webp, svg, bmp, avif, heic, tiff)",
	"`![short description](<absolute source path>)`",
	"Use only the real source path of retrieved evidence",
	"Never invent a path, use relative paths, or embed remote URLs",
	"Apart from these image embeds, the no-file-paths rule stands",
] as const;

function expectImageEmbedRule(text: string): void {
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
	root = mkdtempSync(join(tmpdir(), "autorag-image-embed-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "q3-chart.png"), "not-a-real-png");
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel() {
	const registration = registerFauxProvider({
		api: `faux-image-embed-${randomUUID()}`,
		models: [{ id: "faux-model", reasoning: true }],
	});
	registration.setResponses([fauxAssistantMessage("noop", { stopReason: "stop" })]);
	registrations.push(registration);
	return registration.getModel();
}

function agentOptions(model: ReturnType<typeof fauxModel>): AutoRAGAgentOptions {
	return {
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
	};
}

describe("answer image-embed exception (#1790)", () => {
	it("states the image-embed exception in the system prompt while keeping the no-path rule", () => {
		const prompt = buildSystemPrompt({ toolNames: [], manifests: [], modelId: "test-model" });
		expectImageEmbedRule(prompt);
		// The original ban must survive the exception.
		expect(prompt).toContain("Never include specific file paths");
	});

	it("states the image-embed exception in all three phase prompts", () => {
		const agent = new AutoRAGAgent(agentOptions(fauxModel()));

		const search = agent.buildSearchPrompt("Q3 매출 차트 이미지 보여줘", {});
		const fast = agent.buildFastAnswerPrompt("Q3 매출 차트 이미지 보여줘", {}, "baseline");
		const complete = agent.buildRefinementPrompt("Q3 매출 차트 이미지 보여줘", {}, undefined, false);

		for (const text of [search, fast, complete]) expectImageEmbedRule(text);

		const delta = agent.buildRefinementPrompt(
			"Q3 매출 차트 이미지 보여줘",
			{},
			`![Q3 revenue chart](<${join(docs, "q3-chart.png")}>) [1]`,
			true,
		);
		expectImageEmbedRule(delta);
		// A delta answer must not re-embed an image the first answer already showed.
		expect(delta).toContain("must not be embedded again");
	});

	it("states the image-embed exception in the report answer schema", () => {
		expectImageEmbedRule(answerDescription(reportSchema));
	});
});
