import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";

const [memoryPath, workerId] = process.argv.slice(2);
if (!memoryPath || !workerId || process.send === undefined) {
	throw new Error("memory process fixture requires memory path, worker id, and IPC");
}

const memory = new RetrievalMemory({ storagePath: memoryPath });
memory.load();
memory.recordFeedback(`feedback-${workerId}`, `method-${workerId}`, true);
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
				normalizeSessionEvidenceRef({
					method: `method-${workerId}`,
					source: `/docs/${workerId}.md`,
					excerpt: `Evidence ${workerId}`,
				}),
			],
		},
	],
});

process.send({ type: "ready", workerId });
process.on("message", (message: unknown) => {
	if (message !== "save") return;
	memory.save();
	process.send?.({ type: "saved", workerId }, () => process.disconnect());
});
