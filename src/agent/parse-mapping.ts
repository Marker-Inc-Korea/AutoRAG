import type { CuratedResult } from "../retrieval/types.ts";

export function parseInternalMapping(text: string): CuratedResult[] {
	const match = text.match(/<internal_mapping>([\s\S]*?)<\/internal_mapping>/);
	if (!match) return [];
	const lines = match[1].trim().split("\n");
	const results: CuratedResult[] = [];
	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		const firstColon = trimmed.indexOf(":");
		const lastColon = trimmed.lastIndexOf(":");
		if (firstColon === -1 || firstColon === lastColon) continue;
		const index = Number.parseInt(trimmed.slice(0, firstColon), 10);
		if (Number.isNaN(index)) continue;
		const source = trimmed.slice(firstColon + 1, lastColon);
		const method = trimmed.slice(lastColon + 1);
		results.push({ index, content: "", source, method });
	}
	return results;
}
