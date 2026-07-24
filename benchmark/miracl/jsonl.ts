import { createReadStream } from "node:fs";
import { link, lstat, open, rm, unlink } from "node:fs/promises";
import { createInterface } from "node:readline";
import { Transform, type TransformCallback } from "node:stream";
import type { BenchmarkQuery, Qrel } from "./types.ts";

const MAX_LINE_BYTES = 16 * 1024 * 1024;

class LineLimitTransform extends Transform {
	#bytesSinceNewline = 0;
	#lineNumber = 1;
	readonly #path: string;

	constructor(path: string) {
		super();
		this.#path = path;
	}

	_transform(chunk: Buffer, encoding: BufferEncoding, callback: TransformCallback): void {
		const bytes = typeof chunk === "string" ? Buffer.from(chunk, encoding) : chunk;
		for (const byte of bytes) {
			if (byte === 0x0a) {
				this.#bytesSinceNewline = 0;
				this.#lineNumber += 1;
				continue;
			}
			this.#bytesSinceNewline += 1;
			if (this.#bytesSinceNewline > MAX_LINE_BYTES) {
				callback(new Error(`line ${this.#lineNumber} in ${this.#path} exceeds 16 MiB`));
				return;
			}
		}
		callback(null, chunk);
	}
}

async function forEachLine(path: string, visit: (line: string, lineNumber: number) => void): Promise<void> {
	const input = createReadStream(path, { encoding: "utf8" });
	const lineLimit = new LineLimitTransform(path);
	const reader = createInterface({ input: input.pipe(lineLimit), crlfDelay: Number.POSITIVE_INFINITY });
	let lineNumber = 0;

	try {
		for await (const line of reader) {
			lineNumber += 1;
			visit(line, lineNumber);
		}
	} finally {
		reader.close();
		input.destroy();
		lineLimit.destroy();
	}

	if (lineNumber === 0) {
		throw new Error(`input file ${path} is empty`);
	}
}

function assertNonBlankId(value: string, label: string, path: string, lineNumber: number): void {
	if (value.trim().length === 0) {
		throw new Error(`${label} at ${path}:${lineNumber} must not be blank`);
	}
}

export async function readJsonLines<T>(path: string): Promise<T[]> {
	const records: T[] = [];
	await forEachLine(path, (line, lineNumber) => {
		if (line.trim().length === 0) {
			throw new Error(`JSONL line at ${path}:${lineNumber} must not be blank`);
		}
		try {
			records.push(JSON.parse(line) as T);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			throw new Error(`invalid JSON at ${path}:${lineNumber}: ${message}`);
		}
	});
	return records;
}

export async function readTopicsTsv(path: string): Promise<BenchmarkQuery[]> {
	const topics: BenchmarkQuery[] = [];
	const queryIds = new Set<string>();

	await forEachLine(path, (line, lineNumber) => {
		const fields = line.split("\t");
		if (fields.length !== 2) {
			throw new Error(`topic at ${path}:${lineNumber} must contain exactly two TSV columns`);
		}
		const [queryId, text] = fields;
		assertNonBlankId(queryId, "query id", path, lineNumber);
		if (queryIds.has(queryId)) {
			throw new Error(`duplicate query id ${queryId} at ${path}:${lineNumber}`);
		}
		queryIds.add(queryId);
		topics.push({ queryId, text });
	});

	return topics;
}

export async function readQrels(path: string): Promise<Qrel[]> {
	const qrels: Qrel[] = [];
	const pairs = new Set<string>();

	await forEachLine(path, (line, lineNumber) => {
		const fields = line.trim().split(/\s+/);
		if (fields.length !== 4) {
			throw new Error(`qrel at ${path}:${lineNumber} must contain exactly four columns`);
		}
		const [queryId, , documentId, relevanceText] = fields;
		assertNonBlankId(queryId, "query id", path, lineNumber);
		assertNonBlankId(documentId, "document id", path, lineNumber);

		const relevance = Number(relevanceText);
		if (!Number.isFinite(relevance) || !Number.isInteger(relevance) || relevance < 0) {
			throw new Error(`qrel relevance at ${path}:${lineNumber} must be a finite integer greater than or equal to zero`);
		}

		const pair = `${queryId}\u0000${documentId}`;
		if (pairs.has(pair)) {
			throw new Error(`duplicate qrel for ${queryId}/${documentId} at ${path}:${lineNumber}`);
		}
		pairs.add(pair);
		qrels.push({ queryId, documentId, relevance });
	});

	return qrels;
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
	try {
		await lstat(path);
		throw new Error(`destination already exists: ${path}`);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
			throw error;
		}
	}

	const temporaryPath = `${path}.tmp-${process.pid}`;
	let temporaryFileCreated = false;
	try {
		const serialized = JSON.stringify(value);
		if (serialized === undefined) {
			throw new Error("value is not JSON serializable");
		}
		const handle = await open(temporaryPath, "wx", 0o600);
		temporaryFileCreated = true;
		try {
			await handle.writeFile(`${serialized}\n`, "utf8");
		} finally {
			await handle.close();
		}
		try {
			await link(temporaryPath, path);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code === "EEXIST") {
				throw new Error(`destination already exists: ${path}`);
			}
			throw error;
		}
		await unlink(temporaryPath);
		temporaryFileCreated = false;
	} finally {
		if (temporaryFileCreated) {
			await rm(temporaryPath, { force: true });
		}
	}
}
