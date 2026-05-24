import { describe, expect, it } from "vitest";
import { RetrievalMethodRegistry } from "../../src/retrieval/registry.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor } from "../../src/retrieval/types.ts";

// A minimal mock implementation for testing
function makeMockMethod(name: string, type: "posix" | "vector" | "bm25" | "hybrid" | "visual"): RetrievalMethod {
	return {
		describe(): RetrievalMethodDescriptor {
			return {
				name,
				type,
				description: `Mock ${name} method`,
				status: "active",
				capabilities: ["test"],
			};
		},
		async retrieve(_query, _options) {
			return [];
		},
	};
}

describe("RetrievalMethodRegistry", () => {
	it("starts empty", () => {
		const registry = new RetrievalMethodRegistry();
		expect(registry.list()).toEqual([]);
	});

	it("registers and retrieves a method by name", () => {
		const registry = new RetrievalMethodRegistry();
		const method = makeMockMethod("posix", "posix");
		registry.register(method);
		expect(registry.get("posix")).toBe(method);
	});

	it("lists all registered methods", () => {
		const registry = new RetrievalMethodRegistry();
		registry.register(makeMockMethod("posix", "posix"));
		registry.register(makeMockMethod("vector", "vector"));
		expect(registry.list()).toHaveLength(2);
	});

	it("throws on duplicate registration", () => {
		const registry = new RetrievalMethodRegistry();
		registry.register(makeMockMethod("posix", "posix"));
		expect(() => registry.register(makeMockMethod("posix", "posix"))).toThrow("already registered");
	});

	it("filters methods by type", () => {
		const registry = new RetrievalMethodRegistry();
		registry.register(makeMockMethod("posix", "posix"));
		registry.register(makeMockMethod("vector", "vector"));
		registry.register(makeMockMethod("bm25", "bm25"));
		const vectorMethods = registry.getByType("vector");
		expect(vectorMethods).toHaveLength(1);
		expect(vectorMethods[0].describe().name).toBe("vector");
	});

	it("returns undefined for unknown method name", () => {
		const registry = new RetrievalMethodRegistry();
		expect(registry.get("nonexistent")).toBeUndefined();
	});
});
