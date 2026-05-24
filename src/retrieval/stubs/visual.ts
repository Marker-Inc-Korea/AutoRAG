import { NotImplementedError } from "../../types/errors.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export class VisualRetrieval implements RetrievalMethod {
	describe(): RetrievalMethodDescriptor {
		return {
			name: "visual",
			type: "visual",
			description: "Visual document retrieval using ColPali-style page-level image embeddings",
			status: "stub",
			capabilities: ["visual-search", "colpali", "page-image-embedding", "multimodal"],
		};
	}

	async retrieve(_query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> {
		throw new NotImplementedError("visual");
	}
}
