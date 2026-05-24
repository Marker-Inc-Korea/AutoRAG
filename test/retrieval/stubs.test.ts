import { describe, expect, it } from "vitest";
import { RetrievalMethodRegistry } from "../../src/retrieval/registry.ts";
import { BM25Retrieval } from "../../src/retrieval/stubs/bm25.ts";
import { HybridRetrieval } from "../../src/retrieval/stubs/hybrid.ts";
import { VectorSearchRetrieval } from "../../src/retrieval/stubs/vector.ts";
import { VisualRetrieval } from "../../src/retrieval/stubs/visual.ts";
import { NotImplementedError } from "../../src/types/errors.ts";

const stubs = [
	{ name: "vector", type: "vector", Cls: VectorSearchRetrieval },
	{ name: "bm25", type: "bm25", Cls: BM25Retrieval },
	{ name: "hybrid", type: "hybrid", Cls: HybridRetrieval },
	{ name: "visual", type: "visual", Cls: VisualRetrieval },
] as const;

describe("Retrieval stubs", () => {
	for (const { name, type, Cls } of stubs) {
		it(`${name}: describe() returns correct type and stub status`, () => {
			const stub = new Cls();
			const desc = stub.describe();
			expect(desc.name).toBe(name);
			expect(desc.type).toBe(type);
			expect(desc.status).toBe("stub");
			expect(desc.capabilities.length).toBeGreaterThan(0);
		});

		it(`${name}: retrieve() throws NotImplementedError`, async () => {
			const stub = new Cls();
			await expect(stub.retrieve("test query", {})).rejects.toThrow(NotImplementedError);
			await expect(stub.retrieve("test query", {})).rejects.toThrow(name);
		});

		it(`${name}: can be registered in registry`, () => {
			const registry = new RetrievalMethodRegistry();
			const stub = new Cls();
			expect(() => registry.register(stub)).not.toThrow();
			expect(registry.get(name)).toBe(stub);
		});
	}
});
