import { type EvidenceChunkRecord, RetrievalMemory } from "../../src/memory/memory.ts";

/** The evidence chunks persisted to retrieval memory for one numbered result of a session, in citation order. */
export function storedEvidence(memoryPath: string, sessionId: string, resultNumber: number): EvidenceChunkRecord[] {
	const memory = new RetrievalMemory({ storagePath: memoryPath });
	memory.load();
	const schema = memory.getSchema();
	const result = schema.curatedResults.find((entry) => entry.sessionId === sessionId && entry.number === resultNumber);
	if (result === undefined) return [];
	return result.evidenceIds.flatMap((id) => schema.evidenceChunks.filter((chunk) => chunk.stableEvidenceId === id));
}
