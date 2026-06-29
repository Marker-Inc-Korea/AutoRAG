import { XMLParser } from "fast-xml-parser";
import JSZip from "jszip";

const parser = new XMLParser({
	ignoreAttributes: false,
	parseTagValue: false,
	processEntities: true,
	trimValues: false,
});

export async function readZipXmlText(bytes: Uint8Array, pathPattern: RegExp): Promise<string[]> {
	const zip = await JSZip.loadAsync(bytes);
	const files = Object.values(zip.files)
		.filter((file) => !file.dir && pathPattern.test(file.name))
		.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
	const chunks: string[] = [];
	for (const file of files) {
		const xml = await file.async("text");
		chunks.push(...extractTextFromXml(xml));
	}
	return chunks;
}

export function extractTextFromXml(xml: string): string[] {
	return collectText(parser.parse(xml));
}

function collectText(value: unknown): string[] {
	if (typeof value === "string") {
		const trimmed = value.trim();
		return trimmed.length > 0 ? [trimmed] : [];
	}
	if (Array.isArray(value)) return value.flatMap((item) => collectText(item));
	if (!isRecord(value)) return [];

	const chunks: string[] = [];
	for (const [key, child] of Object.entries(value)) {
		if (key.startsWith("@_")) continue;
		if (isTextElement(key)) {
			chunks.push(...collectText(child));
			continue;
		}
		chunks.push(...collectText(child));
	}
	return chunks;
}

function isTextElement(key: string): boolean {
	return key === "t" || key.endsWith(":t");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
