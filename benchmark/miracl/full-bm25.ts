import { existsSync, lstatSync, mkdirSync, realpathSync, rmSync } from "node:fs";
import { join } from "node:path";
import { normalizeMarkdown } from "../../src/parser/text.ts";
import { chunkBm25Markdown } from "../../src/retrieval/methods/bm25.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";
import { forEachJsonLine, type JsonLinesAttestation } from "./jsonl.ts";
import type { CorpusDocument } from "./types.ts";

type TantivyBinding = typeof import("@pngwasi/node-tantivy-binding");

const TANTIVY_INDEX_PATH = join(".autorag", "bm25", "tantivy");

export interface FullCorpusBm25Options {
	readonly root: string;
	readonly corpusPath: string;
	readonly attestation: JsonLinesAttestation;
	readonly importBinding?: () => Promise<TantivyBinding>;
}

export interface FullCorpusBm25SyncResult {
	readonly engine: "tantivy";
	readonly indexedDocuments: number;
	readonly indexedChunks: number;
}

export interface StreamFullCorpusOptions {
	readonly path: string;
	readonly attestation: JsonLinesAttestation;
	readonly requiredDocumentIds?: ReadonlySet<string>;
	readonly visit?: (document: CorpusDocument, lineNumber: number) => void | Promise<void>;
}

export async function streamFullCorpus(options: StreamFullCorpusOptions): Promise<JsonLinesAttestation> {
	const missingDocumentIds =
		options.requiredDocumentIds === undefined ? undefined : new Set(options.requiredDocumentIds);
	const actual = await forEachJsonLine<unknown>(
		options.path,
		async (value, lineNumber) => {
			const document = validateCorpusDocument(value, lineNumber);
			missingDocumentIds?.delete(document.documentId);
			await options.visit?.(document, lineNumber);
		},
		{
			totalBytes: options.attestation.bytes,
			maxRecords: options.attestation.records,
			requirePrivateFile: true,
			label: "prepared full corpus",
		},
	);
	assertAttestation(actual, options.attestation);
	if (missingDocumentIds !== undefined && missingDocumentIds.size > 0) {
		throw new Error(`qrel references missing prepared document ${missingDocumentIds.values().next().value}`);
	}
	return actual;
}

export class FullCorpusBm25Method implements RetrievalMethod {
	readonly #root: string;
	readonly #corpusPath: string;
	readonly #attestation: JsonLinesAttestation;
	readonly #importBinding: () => Promise<TantivyBinding>;
	#ready = false;

	constructor(options: FullCorpusBm25Options) {
		this.#root = canonicalDirectory(options.root, "full BM25 workspace");
		this.#corpusPath = options.corpusPath;
		this.#attestation = options.attestation;
		this.#importBinding = options.importBinding ?? (() => import("@pngwasi/node-tantivy-binding"));
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "bm25",
			type: "bm25",
			description: "Streaming full-corpus Tantivy BM25 with production chunk and query semantics",
			status: this.#ready ? "active" : "stub",
			capabilities: ["lexical", "chunked", "virtual-paths", "engine:tantivy", "streaming-index"],
		};
	}

	async sync(): Promise<FullCorpusBm25SyncResult> {
		const binding = await this.#importBinding();
		const indexPath = join(this.#root, TANTIVY_INDEX_PATH);
		rmSync(indexPath, { recursive: true, force: true });
		mkdirSync(indexPath, { recursive: true, mode: 0o700 });
		const schema = new binding.SchemaBuilder()
			.addTextField("virtualPath", { stored: true, indexOption: "basic", tokenizerName: "raw" })
			.addTextField("chunkId", { stored: true, indexOption: "basic", tokenizerName: "raw" })
			.addTextField("content", { stored: true, indexOption: "position" })
			.build();
		const index = new binding.Index(schema, indexPath, false);
		const writer = index.writer(30_000_000, 1);
		let indexedDocuments = 0;
		let indexedChunks = 0;
		await streamFullCorpus({
			path: this.#corpusPath,
			attestation: this.#attestation,
			visit: (document) => {
				const virtualPath = `/miracl/${encodeURIComponent(document.documentId)}.md`;
				const markdown = normalizeMarkdown(`# ${document.title}\n\n${document.text}\n`);
				for (const [chunkIndex, content] of chunkBm25Markdown(markdown).entries()) {
					const tantivyDocument = new binding.Document();
					tantivyDocument.addText("virtualPath", virtualPath);
					tantivyDocument.addText("chunkId", String(chunkIndex));
					tantivyDocument.addText("content", content);
					writer.addDocument(tantivyDocument);
					indexedChunks += 1;
				}
				indexedDocuments += 1;
			},
		});
		if (indexedDocuments !== this.#attestation.records || indexedChunks < indexedDocuments) {
			throw new Error("full BM25 corpus indexing did not cover the attested corpus");
		}
		writer.commit();
		this.#ready = true;
		return { engine: "tantivy", indexedDocuments, indexedChunks };
	}

	async retrieve(queryText: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		if (!this.#ready) throw new Error("full BM25 index is not ready");
		const query = queryText.trim();
		if (query.length === 0) return [];
		const topK = options.topK ?? 20;
		if (!Number.isSafeInteger(topK) || topK < 1) throw new Error("full BM25 topK must be a positive safe integer");
		const binding = await this.#importBinding();
		const indexPath = join(this.#root, TANTIVY_INDEX_PATH);
		if (!existsSync(indexPath) || !binding.Index.exists(indexPath)) {
			throw new Error("full BM25 index is missing");
		}
		const index = binding.Index.open(indexPath);
		const searcher = index.searcher();
		const parsedQuery = index.parseQueryLenient(query, ["content"])[0];
		const page = searcher.search(parsedQuery, topK, true);
		return page.hits.flatMap((hit): RetrievalResult[] => {
			const document = searcher.doc(hit.docAddress).toDict() as Record<string, unknown[]>;
			const virtualPath = firstString(document.virtualPath);
			const content = firstString(document.content);
			const chunkId = firstString(document.chunkId) ?? "0";
			if (virtualPath === undefined || content === undefined) return [];
			return [
				{
					id: `bm25:${virtualPath}:${chunkId}`,
					content,
					source: virtualPath,
					score: hit.score ?? 0,
					metadata: {
						method: "bm25",
						chunkIndex: Number(chunkId),
						readiness: "ready",
						engine: "tantivy",
					},
				},
			];
		});
	}
}

function validateCorpusDocument(value: unknown, lineNumber: number): CorpusDocument {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`prepared full corpus record ${lineNumber} must be an object`);
	}
	const record = value as Record<string, unknown>;
	const keys = Object.keys(record);
	if (keys.length !== 3 || !keys.includes("documentId") || !keys.includes("title") || !keys.includes("text")) {
		throw new Error(`prepared full corpus record ${lineNumber} has an invalid shape`);
	}
	if (typeof record.documentId !== "string" || record.documentId.trim().length === 0) {
		throw new Error(`prepared full corpus record ${lineNumber} documentId must be non-blank`);
	}
	if (typeof record.title !== "string" || typeof record.text !== "string") {
		throw new Error(`prepared full corpus record ${lineNumber} title and text must be strings`);
	}
	return { documentId: record.documentId, title: record.title, text: record.text };
}

function assertAttestation(actual: JsonLinesAttestation, expected: JsonLinesAttestation): void {
	if (actual.sha256 !== expected.sha256 || actual.bytes !== expected.bytes || actual.records !== expected.records) {
		throw new Error("prepared full corpus identity does not match manifest");
	}
}

function canonicalDirectory(path: string, label: string): string {
	try {
		const stats = lstatSync(path);
		if (!stats.isDirectory() || stats.isSymbolicLink()) throw new Error("not a directory");
		return realpathSync(path);
	} catch {
		throw new Error(`${label} must be a real directory`);
	}
}

function firstString(value: unknown): string | undefined {
	if (Array.isArray(value)) {
		return value.find((entry): entry is string => typeof entry === "string");
	}
	return typeof value === "string" ? value : undefined;
}
