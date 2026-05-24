import { NotImplementedError } from "../../types/errors.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export class HybridRetrieval implements RetrievalMethod {
	describe(): RetrievalMethodDescriptor {
		return {
			name: "hybrid",
			type: "hybrid",
			description: "Hybrid retrieval combining vector search and BM25 with score fusion",
			status: "stub",
			capabilities: ["hybrid-search", "score-fusion", "rrf", "dense-sparse-combination"],
		};
	}

	async retrieve(_query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> {
		throw new NotImplementedError("hybrid");
	}
}
