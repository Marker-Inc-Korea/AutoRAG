import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

const [memoryPath, workerId] = process.argv.slice(2);
if (!memoryPath || !workerId || process.send === undefined) {
	throw new Error("memory process fixture requires memory path, worker id, and IPC");
}

const stableEvidenceId = `worker:${workerId}`;
const record: JudgedEvidenceRecord = {
	id: `session-${workerId}:${stableEvidenceId}`,
	sessionId: `session-${workerId}`,
	conversationId: `conversation-${workerId}`,
	question: `question ${workerId}`,
	searchQuery: `query ${workerId}`,
	method: `method-${workerId}`,
	source: `/docs/${workerId}.md`,
	stableEvidenceId,
	resultNumber: 1,
	title: `Result ${workerId}`,
	excerpt: `Evidence ${workerId}`,
	probability: 0.9,
	createdAt: 1_000,
};

const memory = new RetrievalMemory({ storagePath: memoryPath });
memory.load();
memory.recordJudgedEvidence([record]);
memory.recordCuratedResultsSession({
	sessionId: `session-${workerId}`,
	query: `session query ${workerId}`,
	results: [
		{
			number: 1,
			title: `Result ${workerId}`,
			summary: `Summary ${workerId}`,
			content: `Content ${workerId}`,
			method: `method-${workerId}`,
			source: `/docs/${workerId}.md`,
			evidenceRefs: [
				{
					method: `method-${workerId}`,
					source: `/docs/${workerId}.md`,
					content: `Evidence ${workerId}`,
					stableEvidenceId,
				},
			],
		},
	],
});

process.send({ type: "ready", workerId });
process.on("message", (message: unknown) => {
	if (message !== "save") return;
	memory.save();
	process.send?.({ type: "saved", workerId }, () => process.disconnect?.());
});
