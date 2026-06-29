import { Buffer } from "node:buffer";
import { XMLParser } from "fast-xml-parser";
import JSZip from "jszip";

const parser = new XMLParser({
	ignoreAttributes: false,
	parseTagValue: false,
	processEntities: true,
	trimValues: false,
});

const MAX_XML_FILES = 64;
const MAX_XML_BYTES = 5_000_000;
const MAX_TEXT_CHUNKS = 20_000;

export async function readZipXmlText(bytes: Uint8Array, pathPattern: RegExp): Promise<string[]> {
	const zip = await JSZip.loadAsync(bytes);
	const files = Object.values(zip.files)
		.filter((file) => !file.dir && pathPattern.test(file.name))
		.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
	if (files.length > MAX_XML_FILES) {
		throw new Error(`archive XML file count exceeds limit of ${MAX_XML_FILES}`);
	}
	const chunks: string[] = [];
	let totalXmlBytes = 0;
	for (const file of files) {
		const declaredSize = uncompressedSize(file);
		if (declaredSize !== undefined && totalXmlBytes + declaredSize > MAX_XML_BYTES) {
			throw new Error(`archive XML content exceeds limit of ${MAX_XML_BYTES} bytes`);
		}
		const xml = await readXmlFile(file, MAX_XML_BYTES - totalXmlBytes);
		totalXmlBytes += Buffer.byteLength(xml);
		if (totalXmlBytes > MAX_XML_BYTES) {
			throw new Error(`archive XML content exceeds limit of ${MAX_XML_BYTES} bytes`);
		}
		chunks.push(...extractTextFromXml(xml));
		if (chunks.length > MAX_TEXT_CHUNKS) {
			throw new Error(`archive text chunk count exceeds limit of ${MAX_TEXT_CHUNKS}`);
		}
	}
	return chunks;
}

async function readXmlFile(file: JSZip.JSZipObject, remainingBytes: number): Promise<string> {
	const stream = file.nodeStream("nodebuffer");
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		let totalBytes = 0;
		stream.on("data", (chunk: unknown) => {
			const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as ArrayBuffer);
			totalBytes += buffer.byteLength;
			if (totalBytes > remainingBytes) {
				destroyStream(stream);
				reject(new Error(`archive XML content exceeds limit of ${MAX_XML_BYTES} bytes`));
				return;
			}
			chunks.push(buffer);
		});
		stream.on("error", reject);
		stream.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
	});
}

function uncompressedSize(file: JSZip.JSZipObject): number | undefined {
	const data = Reflect.get(file, "_data");
	if (!isRecord(data)) return undefined;
	const size = data.uncompressedSize;
	return typeof size === "number" ? size : undefined;
}

function destroyStream(stream: NodeJS.ReadableStream): void {
	if ("destroy" in stream && typeof stream.destroy === "function") {
		stream.destroy();
	}
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
