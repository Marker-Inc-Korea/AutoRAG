import { NotImplementedError } from "../../types/errors.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export class BM25Retrieval implements RetrievalMethod {
	describe(): RetrievalMethodDescriptor {
		return {
			name: "bm25",
			type: "bm25",
			description: "BM25 keyword-based retrieval using TF-IDF scoring",
			status: "stub",
			capabilities: ["keyword-search", "bm25-scoring", "tf-idf"],
		};
	}

	async retrieve(_query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> {
		throw new NotImplementedError("bm25");
	}
}
