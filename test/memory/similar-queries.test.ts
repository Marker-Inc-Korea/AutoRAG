import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Embedder, EmbeddingIdentity } from "../../src/embedding-runtime/gateway-embedder.ts";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import {
	findSimilarQuestions,
	memoryVectorStorePath,
	SIMILAR_QUESTION_LIMIT,
} from "../../src/memory/similar-queries.ts";

let tmpDir: string;
let vectorPath: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-similar-queries-"));
	vectorPath = join(tmpDir, "memory.json.vectors.db");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

let counter = 0;

function record(question: string, overrides: Partial<JudgedEvidenceRecord> = {}): JudgedEvidenceRecord {
	counter += 1;
	const sessionId = overrides.sessionId ?? `session-${counter}`;
	return {
		id: `${sessionId}:method:e${counter}`,
		sessionId,
		conversationId: "past-conversation",
		question,
		searchQuery: question,
		method: "bash",
		source: `/docs/${counter}.md`,
		stableEvidenceId: `method:e${counter}`,
		resultNumber: 1,
		title: `Title ${counter}`,
		excerpt: `Excerpt ${counter}`,
		probability: 0.8,
		createdAt: counter,
		...overrides,
	};
}

/** Embedder over a fixed text -> vector table; records every batch it was asked to embed. */
function tableEmbedder(
	table: Readonly<Record<string, readonly number[]>>,
	identity: EmbeddingIdentity = { provider: "test", model: "table", dimension: 2 },
): Embedder & { readonly calls: string[][] } {
	const calls: string[][] = [];
	return {
		calls,
		identity: async () => identity,
		embed: async (texts) => {
			calls.push([...texts]);
			return texts.map((text) => {
				const vector = table[text];
				if (vector === undefined) throw new Error(`no vector for ${JSON.stringify(text)}`);
				return [...vector];
			});
		},
	};
}

describe("memoryVectorStorePath", () => {
	it("sits next to the memory file", () => {
		expect(memoryVectorStorePath("/home/u/.autorag/memory.json")).toBe("/home/u/.autorag/memory.json.vectors.db");
	});
});

describe("findSimilarQuestions without an embedder (BM25 only)", () => {
	it("ranks past questions by lexical overlap and ignores unrelated ones", async () => {
		const records = [
			record("who approved the Q3 budget memo"),
			record("Q3 budget memo approval history"),
			record("how to bake sourdough bread"),
		];
		const result = await findSimilarQuestions(records, "who approved the Q3 budget");
		expect(result.fallbackReason).toBeUndefined();
		expect(result.questions.map((entry) => entry.question)).toEqual([
			"who approved the Q3 budget memo",
			"Q3 budget memo approval history",
		]);
	});

	it("groups every judged record of one question, best evidence first", async () => {
		const weak = record("budget memo owner", { sessionId: "s1", probability: 0.71, title: "weak" });
		const strong = record("budget memo owner", { sessionId: "s2", probability: 0.95, title: "strong" });
		const result = await findSimilarQuestions([weak, strong], "budget memo owner");
		expect(result.questions).toHaveLength(1);
		expect(result.questions[0]?.records.map((entry) => entry.title)).toEqual(["strong", "weak"]);
	});

	it("returns at most 25 questions by default and honours a smaller limit", async () => {
		const records = Array.from({ length: 40 }, (_, index) => record(`budget memo variant ${index}`));
		const defaulted = await findSimilarQuestions(records, "budget memo");
		expect(defaulted.questions).toHaveLength(SIMILAR_QUESTION_LIMIT);
		expect(SIMILAR_QUESTION_LIMIT).toBe(25);
		const limited = await findSimilarQuestions(records, "budget memo", { limit: 3 });
		expect(limited.questions).toHaveLength(3);
	});

	it("matches Korean particles by prefix", async () => {
		const result = await findSimilarQuestions([record("인증서를 발급하는 방법")], "인증서 발급");
		expect(result.questions.map((entry) => entry.question)).toEqual(["인증서를 발급하는 방법"]);
	});

	it("returns nothing for an empty memory or an empty query", async () => {
		expect((await findSimilarQuestions([], "anything")).questions).toEqual([]);
		expect((await findSimilarQuestions([record("budget")], "   ")).questions).toEqual([]);
	});
});

describe("findSimilarQuestions with an embedder (BM25 + vector)", () => {
	const table = {
		"current question": [1, 0],
		"lexical match only": [0, 1],
		"semantic match only": [0.99, 0.1],
		"both lexical and semantic": [1, 0.05],
		unrelated: [0, 1],
	};

	it("returns a semantically close question that shares no word with the query", async () => {
		const embedder = tableEmbedder({ ...table, "current question": [1, 0], "semantic match only": [0.99, 0.1] });
		const result = await findSimilarQuestions(
			[record("semantic match only"), record("unrelated")],
			"current question",
			{ embedder, vectorStorePath: vectorPath },
		);
		expect(result.questions.map((entry) => entry.question)).toEqual(["semantic match only"]);
	});

	it("ranks a question found by both searches above one found by a single search", async () => {
		const embedder = tableEmbedder({
			"current question": [1, 0],
			"current question both": [1, 0.05],
			"current question lexical": [0, 1],
			"semantic only": [0.95, 0.2],
		});
		const result = await findSimilarQuestions(
			[record("semantic only"), record("current question lexical"), record("current question both")],
			"current question",
			{ embedder, vectorStorePath: vectorPath },
		);
		expect(result.questions[0]?.question).toBe("current question both");
		expect(result.questions.map((entry) => entry.question)).toContain("semantic only");
	});

	it("persists question vectors and embeds only what is new on the next lookup", async () => {
		const embedder = tableEmbedder(table);
		const records = [record("semantic match only")];
		await findSimilarQuestions(records, "current question", { embedder, vectorStorePath: vectorPath });
		expect(existsSync(vectorPath)).toBe(true);
		expect(embedder.calls.flat().sort()).toEqual(["current question", "semantic match only"]);

		embedder.calls.length = 0;
		await findSimilarQuestions(records, "current question", { embedder, vectorStorePath: vectorPath });
		expect(embedder.calls.flat()).toEqual(["current question"]);
	});

	it("discards stored vectors made by a different embedding model", async () => {
		const first = tableEmbedder(table, { provider: "test", model: "old", dimension: 2 });
		const records = [record("semantic match only")];
		await findSimilarQuestions(records, "current question", { embedder: first, vectorStorePath: vectorPath });

		const second = tableEmbedder(table, { provider: "test", model: "new", dimension: 2 });
		await findSimilarQuestions(records, "current question", { embedder: second, vectorStorePath: vectorPath });
		expect(second.calls.flat().sort()).toEqual(["current question", "semantic match only"]);
		second.calls.length = 0;
		await findSimilarQuestions(records, "current question", { embedder: second, vectorStorePath: vectorPath });
		expect(second.calls.flat()).toEqual(["current question"]);
	});

	it("keeps the vectors of concurrent lookups that each embed different questions", async () => {
		const embedder = tableEmbedder({ ...table, "other current": [0.9, 0.1], "other semantic": [0.8, 0.3] });
		await Promise.all([
			findSimilarQuestions([record("semantic match only")], "current question", {
				embedder,
				vectorStorePath: vectorPath,
			}),
			findSimilarQuestions([record("other semantic")], "other current", { embedder, vectorStorePath: vectorPath }),
		]);

		embedder.calls.length = 0;
		await findSimilarQuestions([record("semantic match only"), record("other semantic")], "current question", {
			embedder,
			vectorStorePath: vectorPath,
		});
		expect(embedder.calls.flat()).toEqual(["current question"]);
	});

	it("falls back to BM25 and reports the embedder's own error text verbatim", async () => {
		const embedder: Embedder = {
			identity: async () => ({ provider: "test", model: "down", dimension: 2 }),
			embed: async () => {
				throw new Error("gateway /v1/embeddings returned HTTP 503");
			},
		};
		const result = await findSimilarQuestions([record("current question details")], "current question", {
			embedder,
			vectorStorePath: vectorPath,
		});
		expect(result.questions.map((entry) => entry.question)).toEqual(["current question details"]);
		expect(result.fallbackReason).toContain("gateway /v1/embeddings returned HTTP 503");
	});

	it("reports an unreadable vector store but keeps searching", async () => {
		writeFileSync(vectorPath, "this is not a sqlite database, just text\n".repeat(200));
		const embedder = tableEmbedder(table);
		const result = await findSimilarQuestions([record("semantic match only")], "current question", {
			embedder,
			vectorStorePath: vectorPath,
		});
		expect(result.questions.map((entry) => entry.question)).toEqual(["semantic match only"]);

		embedder.calls.length = 0;
		await findSimilarQuestions([record("semantic match only")], "current question", {
			embedder,
			vectorStorePath: vectorPath,
		});
		expect(embedder.calls.flat()).toEqual(["current question"]);
	});
});
