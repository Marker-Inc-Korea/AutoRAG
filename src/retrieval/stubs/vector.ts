import { NotImplementedError } from "../../types/errors.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export class VectorSearchRetrieval implements RetrievalMethod {
	describe(): RetrievalMethodDescriptor {
		return {
			name: "vector",
			type: "vector",
			description: "Semantic vector similarity search using dense embeddings",
			status: "stub",
			capabilities: ["semantic-search", "dense-retrieval", "embedding-similarity"],
		};
	}

	async retrieve(_query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> {
		throw new NotImplementedError("vector");
	}
}
