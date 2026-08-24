import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadLocalAutoRAGModels } from "../../src/subagents/local-models.ts";
import { createMandatorySubagentSession } from "../../src/subagents/runtime.ts";

const root = mkdtempSync(join(tmpdir(), "autorag-issue-1455-live-"));
const docs = join(root, "docs");
const agentDir = join(root, "pi-agent");
const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
mkdirSync(docs, { recursive: true });
writeFileSync(
	join(docs, "policy.txt"),
	[
		"Atlas refund policy",
		"Refund exceptions require director approval before payout.",
		"The policy was revised on 2026-07-01.",
	].join("\n"),
);
delete process.env.PI_CODING_AGENT_DIR;

try {
	const models = loadLocalAutoRAGModels();
	const runtime = await createMandatorySubagentSession({
		cwd: docs,
		agentDir,
		model: models.orchestrator,
		explorerModel: models.explorer,
		apiKey: models.apiKey,
		providerApiKeys: { [models.provider]: models.apiKey },
		systemPrompt: "Run the requested explorer task.",
		tools: [],
	});

	try {
		if (process.env.PI_CODING_AGENT_DIR !== agentDir) {
			throw new Error(`Expected active agent directory ${agentDir}, got ${process.env.PI_CODING_AGENT_DIR}`);
		}
		const subagent = runtime.session.getToolDefinition("subagent");
		if (subagent === undefined) throw new Error("The mandatory subagent tool was not registered");

		const result = await subagent.execute(
			"issue-1455-live",
			{
				agent: "autorag-explorer",
				agentScope: "user",
				artifacts: false,
				model: `${models.provider}/${models.explorer.id}`,
				cwd: docs,
				task: [
					"<<<AUTORAG_ASSIGNMENT_V1>>>",
					JSON.stringify({
						originalQuery: "What approval is required for Atlas refund exceptions?",
						method: "contained read",
						queryVariants: ["Atlas refund exception approval", "refund policy approver"],
					}),
					"<<<END_AUTORAG_ASSIGNMENT_V1>>>",
					"Required handoff: include retrievedAt and temporal metadata.",
					"Read policy.txt and return candidate evidence only.",
				].join("\n"),
			},
			undefined,
			undefined,
			runtime.session.extensionRunner.createContext(),
		);
		const serialized = JSON.stringify(result);
		const results = (result.details as { results?: Array<{ exitCode?: number }> } | undefined)?.results ?? [];
		if (result.isError || serialized.includes("Unknown agent: autorag-explorer")) {
			throw new Error(`Explorer discovery failed: ${serialized}`);
		}
		if (!results.some((entry) => entry.exitCode === 0)) {
			throw new Error(`Explorer did not complete successfully: ${serialized}`);
		}
		if (!serialized.toLowerCase().includes("director approval")) {
			throw new Error(`Explorer did not return grounded policy evidence: ${serialized}`);
		}
		console.log(
			JSON.stringify(
				{
					status: "LIVE E2E PASSED",
					agent: "autorag-explorer",
					model: `${models.provider}/${models.explorer.id}`,
					exitCode: 0,
				},
				null,
				2,
			),
		);
	} finally {
		runtime.session.dispose();
	}

	if (process.env.PI_CODING_AGENT_DIR !== undefined) {
		throw new Error(`Session did not restore PI_CODING_AGENT_DIR: ${process.env.PI_CODING_AGENT_DIR}`);
	}
} finally {
	if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
	else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
	rmSync(root, { recursive: true, force: true });
}
