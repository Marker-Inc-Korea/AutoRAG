export interface StoreManifest {
	name: string;
	description: string;
	type: "vector" | "bm25" | "hybrid" | "visual";
	dataDescription: string;
	contentTypes: string[];
	config: Record<string, unknown>;
}
