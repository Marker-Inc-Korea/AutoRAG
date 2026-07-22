import type { RetrievalMethod } from "./types.ts";

export class RetrievalMethodRegistry {
	private readonly methods: Map<string, RetrievalMethod> = new Map();

	register(method: RetrievalMethod): void {
		const name = method.describe().name;
		if (this.methods.has(name)) {
			throw new Error(`Retrieval method "${name}" is already registered`);
		}
		this.methods.set(name, method);
	}

	get(name: string): RetrievalMethod | undefined {
		return this.methods.get(name);
	}

	list(): RetrievalMethod[] {
		return Array.from(this.methods.values());
	}

	getByType(type: "posix" | "vector" | "bm25" | "hybrid" | "visual"): RetrievalMethod[] {
		return this.list().filter((m) => m.describe().type === type);
	}
}
