import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { link, lstat, open, rm, unlink } from "node:fs/promises";
import { createInterface } from "node:readline";
import { Transform, type TransformCallback } from "node:stream";
import type { BenchmarkQuery, Qrel } from "./types.ts";

const MAX_LINE_BYTES = 16 * 1024 * 1024;

export interface ReadJsonLinesOptions {
	readonly totalBytes?: number;
	readonly maxRecords?: number;
	readonly requirePrivateFile?: boolean;
	readonly label?: string;
}

export interface JsonLinesAttestation {
	readonly sha256: string;
	readonly bytes: number;
	readonly records: number;
}

class LineLimitTransform extends Transform {
	#bytesSinceNewline = 0;
	#lineNumber = 1;
	#totalBytes = 0;
	readonly #path: string;
	readonly #maxTotalBytes: number | undefined;
	readonly #label: string;
	readonly #hash = createHash("sha256");

	constructor(path: string, options: ReadJsonLinesOptions) {
		super();
		this.#path = path;
		this.#maxTotalBytes = options.totalBytes;
		this.#label = options.label ?? `input file ${path}`;
	}

	_transform(chunk: Buffer, encoding: BufferEncoding, callback: TransformCallback): void {
		const bytes = typeof chunk === "string" ? Buffer.from(chunk, encoding) : chunk;
		this.#totalBytes += bytes.byteLength;
		this.#hash.update(bytes);
		if (this.#maxTotalBytes !== undefined && this.#totalBytes > this.#maxTotalBytes) {
			callback(new Error(`${this.#label} exceeds ${this.#maxTotalBytes} bytes`));
			return;
		}
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

	attestation(records: number): JsonLinesAttestation {
		return {
			sha256: this.#hash.digest("hex"),
			bytes: this.#totalBytes,
			records,
		};
	}
}

async function forEachLine(
	path: string,
	visit: (line: string, lineNumber: number) => void | Promise<void>,
	options: ReadJsonLinesOptions = {},
): Promise<JsonLinesAttestation> {
	assertOptionalLimit(options.totalBytes, "totalBytes");
	assertOptionalLimit(options.maxRecords, "maxRecords");
	let exactHandle: Awaited<ReturnType<typeof open>> | undefined;
	let input: ReturnType<typeof createReadStream>;
	if (options.totalBytes !== undefined || options.requirePrivateFile === true) {
		const pathStats = await lstat(path);
		assertBoundedInputStats(pathStats, options, path);
		exactHandle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
		try {
			const openStats = await exactHandle.stat();
			assertBoundedInputStats(openStats, options, path);
			if (openStats.dev !== pathStats.dev || openStats.ino !== pathStats.ino || openStats.size !== pathStats.size) {
				throw new Error(`${options.label ?? `input file ${path}`} changed before reading`);
			}
			input = exactHandle.createReadStream({ encoding: "utf8", autoClose: false });
		} catch (error) {
			await exactHandle.close();
			throw error;
		}
	} else {
		input = createReadStream(path, { encoding: "utf8" });
	}
	const lineLimit = new LineLimitTransform(path, options);
	const reader = createInterface({ input: input.pipe(lineLimit), crlfDelay: Number.POSITIVE_INFINITY });
	let lineNumber = 0;

	try {
		for await (const line of reader) {
			lineNumber += 1;
			if (options.maxRecords !== undefined && lineNumber > options.maxRecords) {
				throw new Error(
					`${options.label ?? `input file ${path}`} must contain at most ${options.maxRecords} records`,
				);
			}
			await visit(line, lineNumber);
		}
		if (exactHandle !== undefined) {
			const finalStats = await exactHandle.stat();
			const pathStats = await lstat(path);
			assertBoundedInputStats(finalStats, options, path);
			assertBoundedInputStats(pathStats, options, path);
			if (
				finalStats.dev !== pathStats.dev ||
				finalStats.ino !== pathStats.ino ||
				finalStats.size !== pathStats.size
			) {
				throw new Error(`${options.label ?? `input file ${path}`} changed while reading`);
			}
		}
	} finally {
		reader.close();
		input.destroy();
		lineLimit.destroy();
		await exactHandle?.close().catch(() => undefined);
	}

	if (lineNumber === 0) {
		throw new Error(`input file ${path} is empty`);
	}
	return lineLimit.attestation(lineNumber);
}

function assertOptionalLimit(value: number | undefined, label: string): void {
	if (value !== undefined && (!Number.isSafeInteger(value) || value < 1)) {
		throw new Error(`${label} must be a positive safe integer`);
	}
}

function assertBoundedInputStats(
	stats: Awaited<ReturnType<typeof lstat>>,
	options: ReadJsonLinesOptions,
	path: string,
): void {
	const label = options.label ?? `input file ${path}`;
	if (!stats.isFile() || stats.isSymbolicLink()) throw new Error(`${label} must be a real file`);
	if (options.requirePrivateFile === true && (Number(stats.mode) & 0o077) !== 0) {
		throw new Error(`${label} must be private`);
	}
	if (options.totalBytes !== undefined && stats.size > options.totalBytes) {
		throw new Error(`${label} exceeds ${options.totalBytes} bytes`);
	}
}

function assertNonBlankId(value: string, label: string, path: string, lineNumber: number): void {
	if (value.trim().length === 0) {
		throw new Error(`${label} at ${path}:${lineNumber} must not be blank`);
	}
}

export async function forEachJsonLine<T>(
	path: string,
	visit: (record: T, lineNumber: number) => void | Promise<void>,
	options: ReadJsonLinesOptions = {},
): Promise<JsonLinesAttestation> {
	return forEachLine(
		path,
		(line, lineNumber) => {
			if (line.trim().length === 0) {
				throw new Error(`JSONL line at ${path}:${lineNumber} must not be blank`);
			}
			try {
				return visit(JSON.parse(line) as T, lineNumber);
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				throw new Error(`invalid JSON at ${path}:${lineNumber}: ${message}`);
			}
		},
		options,
	);
}

export async function readJsonLines<T>(path: string, options: ReadJsonLinesOptions = {}): Promise<T[]> {
	const records: T[] = [];
	await forEachJsonLine<T>(
		path,
		(record) => {
			records.push(record);
		},
		options,
	);
	return records;
}

export async function readTopicsTsv(path: string, options: ReadJsonLinesOptions = {}): Promise<BenchmarkQuery[]> {
	const topics: BenchmarkQuery[] = [];
	const queryIds = new Set<string>();

	await forEachLine(
		path,
		(line, lineNumber) => {
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
		},
		options,
	);

	return topics;
}

export async function readQrels(path: string, options: ReadJsonLinesOptions = {}): Promise<Qrel[]> {
	const qrels: Qrel[] = [];
	const pairs = new Set<string>();

	await forEachLine(
		path,
		(line, lineNumber) => {
			const fields = line.trim().split(/\s+/);
			if (fields.length !== 4) {
				throw new Error(`qrel at ${path}:${lineNumber} must contain exactly four columns`);
			}
			const [queryId, , documentId, relevanceText] = fields;
			assertNonBlankId(queryId, "query id", path, lineNumber);
			assertNonBlankId(documentId, "document id", path, lineNumber);

			const relevance = Number(relevanceText);
			if (!Number.isFinite(relevance) || !Number.isInteger(relevance) || relevance < 0) {
				throw new Error(
					`qrel relevance at ${path}:${lineNumber} must be a finite integer greater than or equal to zero`,
				);
			}

			const pair = `${queryId}\u0000${documentId}`;
			if (pairs.has(pair)) {
				throw new Error(`duplicate qrel for ${queryId}/${documentId} at ${path}:${lineNumber}`);
			}
			pairs.add(pair);
			qrels.push({ queryId, documentId, relevance });
		},
		options,
	);

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
