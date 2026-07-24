import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import type { FileHandle } from "node:fs/promises";
import { lstat, mkdir, open, readdir, rmdir, unlink } from "node:fs/promises";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { Transform, type TransformCallback } from "node:stream";
import { createGunzip } from "node:zlib";
import { readQrels, readTopicsTsv, writeJsonAtomic } from "./jsonl.ts";
import {
	MIRACL_FULL_CORPUS_PASSAGES,
	MIRACL_NORMALIZATION_VERSION,
	MIRACL_SMOKE_PROFILE,
	MIRACL_SOURCES,
} from "./profiles.ts";
import type { BenchmarkProfile, BenchmarkQuery, CorpusDocument, Qrel } from "./types.ts";

const DEFAULT_MAX_REDIRECTS = 5;
const DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024;
const DEFAULT_MAX_DECOMPRESSED_BYTES_PER_SHARD = 4 * 1024 * 1024 * 1024;
const DEFAULT_FETCH_TIMEOUT_MS = 10 * 60 * 1000;
const MAX_CORPUS_LINE_BYTES = 16 * 1024 * 1024;
const MAX_TOPIC_RECORDS = 10_000;
const MAX_QREL_RECORDS = 100_000;
const DUPLICATE_PARTITION_COUNT = 256;
const MAX_DUPLICATE_PARTITION_IDS = 50_000;
const MAX_DUPLICATE_PARTITION_BYTES = 64 * 1024 * 1024;
const MAX_OPEN_DUPLICATE_PARTITIONS = 32;
const DUPLICATE_WRITE_BUFFER_BYTES = 64 * 1024;
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export interface MiraclDataset {
	queries: readonly BenchmarkQuery[];
	qrels: readonly Qrel[];
	corpus: readonly CorpusDocument[];
}

export interface SmokeSelectionOptions {
	seed: number;
	queryCount: number;
	distractorCount: number;
}

interface CommonPrepareOptions {
	outputDir: string;
	fetchImpl?: FetchLike;
	maxRedirects?: number;
	maxDownloadBytes?: number;
	maxDecompressedBytesPerShard?: number;
	fetchTimeoutMs?: number;
}

export interface SmokePrepareOptions extends CommonPrepareOptions, Partial<SmokeSelectionOptions> {
	profile?: "smoke";
}

export interface FullPrepareOptions extends CommonPrepareOptions {
	profile: "full";
}

export type PrepareOptions = SmokePrepareOptions | FullPrepareOptions;

export interface PrepareValidationOptions {
	readonly expectedFullCorpusPassages?: number;
}

export interface PreparedSource {
	url: string;
	path: string;
	sha256: string;
	bytes: number;
}

interface PreparedManifestBase {
	schemaVersion: 1;
	normalizationVersion: typeof MIRACL_NORMALIZATION_VERSION;
	profile: BenchmarkProfile;
	revisions: {
		topics: string;
		corpus: string;
	};
	sources: {
		topics: PreparedSource;
		qrels: PreparedSource;
		corpus: PreparedSource[];
	};
	files: {
		queries: "queries.jsonl";
		qrels: "qrels.jsonl";
		corpus: "corpus.jsonl";
	};
	normalized: {
		queries: NormalizedPreparedFile;
		qrels: NormalizedPreparedFile;
		corpus: NormalizedPreparedFile;
	};
}

export interface SmokePreparedManifest extends PreparedManifestBase {
	profile: "smoke";
	seed: number;
	selectedIds: {
		queryIds: string[];
		documentIds: string[];
	};
	counts: {
		queries: number;
		qrels: number;
		positiveQrels: number;
		corpus: number;
		judgedDocuments: number;
		distractors: number;
	};
}

export interface NormalizedPreparedFile {
	sha256: string;
	bytes: number;
	records: number;
}

export interface FullPreparedManifest extends PreparedManifestBase {
	profile: "full";
	counts: {
		queries: number;
		qrels: number;
		positiveQrels: number;
		corpus: number;
		judgedDocuments: number;
	};
}

export type PreparedManifest = SmokePreparedManifest | FullPreparedManifest;

type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

interface DownloadResult extends PreparedSource {}

interface RankedDocument {
	key: string;
	document: CorpusDocument;
}

interface PreparedDirectoryIdentity {
	readonly device: number;
	readonly inode: number;
}

interface PreparedFileIdentity extends PreparedDirectoryIdentity {
	readonly mode: number;
}

interface PreparedTreeOwnership {
	readonly root: PreparedDirectoryIdentity;
	readonly files: ReadonlyMap<string, PreparedDirectoryIdentity>;
	readonly directories: ReadonlyMap<string, PreparedDirectoryIdentity>;
}

function assertNonBlank(value: string, label: string): void {
	if (typeof value !== "string" || value.trim().length === 0) {
		throw new Error(`${label} must not be blank`);
	}
}

function assertCount(value: number, label: string, allowZero: boolean): void {
	const minimum = allowZero ? 0 : 1;
	if (!Number.isSafeInteger(value) || value < minimum) {
		throw new Error(`${label} must be a safe integer greater than or equal to ${minimum}`);
	}
}

function deterministicKey(seed: number, id: string): string {
	return createHash("sha256").update(`${seed}\0${id}`, "utf8").digest("hex");
}

function compareLexically(left: string, right: string): number {
	return left < right ? -1 : left > right ? 1 : 0;
}

function compareRanked(left: RankedDocument, right: RankedDocument): number {
	const keyOrder = compareLexically(left.key, right.key);
	return keyOrder === 0 ? compareLexically(left.document.documentId, right.document.documentId) : keyOrder;
}

function compareIds(seed: number): (left: string, right: string) => number {
	return (left, right) => {
		const keyOrder = compareLexically(deterministicKey(seed, left), deterministicKey(seed, right));
		return keyOrder === 0 ? compareLexically(left, right) : keyOrder;
	};
}

function validateQueriesAndQrels(
	queries: readonly BenchmarkQuery[],
	qrels: readonly Qrel[],
): Map<string, BenchmarkQuery> {
	const queryById = new Map<string, BenchmarkQuery>();
	for (const query of queries) {
		assertNonBlank(query.queryId, "query id");
		if (queryById.has(query.queryId)) {
			throw new Error(`duplicate query id: ${query.queryId}`);
		}
		queryById.set(query.queryId, query);
	}

	const qrelPairs = new Set<string>();
	for (const qrel of qrels) {
		assertNonBlank(qrel.queryId, "qrel query id");
		assertNonBlank(qrel.documentId, "qrel document id");
		if (!Number.isInteger(qrel.relevance) || !Number.isFinite(qrel.relevance) || qrel.relevance < 0) {
			throw new Error(`qrel relevance for ${qrel.queryId}/${qrel.documentId} must be a non-negative integer`);
		}
		if (!queryById.has(qrel.queryId)) {
			throw new Error(`qrel references missing query: ${qrel.queryId}`);
		}
		const pair = `${qrel.queryId}\0${qrel.documentId}`;
		if (qrelPairs.has(pair)) {
			throw new Error(`duplicate qrel: ${qrel.queryId}/${qrel.documentId}`);
		}
		qrelPairs.add(pair);
	}
	return queryById;
}

function validateDataset(input: MiraclDataset): {
	queryById: Map<string, BenchmarkQuery>;
	documentById: Map<string, CorpusDocument>;
} {
	const queryById = validateQueriesAndQrels(input.queries, input.qrels);

	const documentById = new Map<string, CorpusDocument>();
	for (const document of input.corpus) {
		assertNonBlank(document.documentId, "document id");
		if (documentById.has(document.documentId)) {
			throw new Error(`duplicate document id: ${document.documentId}`);
		}
		documentById.set(document.documentId, document);
	}

	for (const qrel of input.qrels) {
		if (!documentById.has(qrel.documentId)) {
			throw new Error(`qrel references missing corpus document: ${qrel.documentId}`);
		}
	}

	return { queryById, documentById };
}

function validateSelectionOptions(options: SmokeSelectionOptions): void {
	if (!Number.isSafeInteger(options.seed)) {
		throw new Error("seed must be a safe integer");
	}
	assertCount(options.queryCount, "queryCount", false);
	assertCount(options.distractorCount, "distractorCount", true);
}

function selectQueriesAndQrels(
	queries: readonly BenchmarkQuery[],
	qrels: readonly Qrel[],
	options: SmokeSelectionOptions,
): {
	queries: BenchmarkQuery[];
	qrels: Qrel[];
} {
	const queryById = validateQueriesAndQrels(queries, qrels);
	const positiveQueryIds = new Set(qrels.filter((qrel) => qrel.relevance > 0).map((qrel) => qrel.queryId));
	if (positiveQueryIds.size < options.queryCount) {
		throw new Error(
			`requested ${options.queryCount} queries but only ${positiveQueryIds.size} have positive relevance judgments`,
		);
	}

	const compare = compareIds(options.seed);
	const selectedQueryIds = [...positiveQueryIds].sort(compare).slice(0, options.queryCount);
	const selectedQueryIdSet = new Set(selectedQueryIds);
	return {
		queries: selectedQueryIds.map((queryId) => queryById.get(queryId) as BenchmarkQuery),
		qrels: qrels
			.filter((qrel) => selectedQueryIdSet.has(qrel.queryId))
			.sort((left, right) => {
				const queryOrder = compare(left.queryId, right.queryId);
				return queryOrder === 0 ? compare(left.documentId, right.documentId) : queryOrder;
			}),
	};
}

export function selectSmokeDataset(input: MiraclDataset, options: SmokeSelectionOptions): MiraclDataset {
	validateSelectionOptions(options);

	const { queryById, documentById } = validateDataset(input);
	const selected = selectQueriesAndQrels(input.queries, input.qrels, options);
	const compare = compareIds(options.seed);
	const judgedDocumentIds = new Set(selected.qrels.map((qrel) => qrel.documentId));
	const distractors = new BoundedDistractorHeap(options.distractorCount);
	for (const document of documentById.values()) {
		if (!judgedDocumentIds.has(document.documentId)) {
			distractors.add({ key: deterministicKey(options.seed, document.documentId), document });
		}
	}
	const selectedDistractors = distractors.values();
	if (selectedDistractors.length < options.distractorCount) {
		throw new Error(
			`requested ${options.distractorCount} distractors but only ${selectedDistractors.length} unjudged documents are available`,
		);
	}
	const selectedDocumentIds = [
		...judgedDocumentIds,
		...selectedDistractors.map((document) => document.documentId),
	].sort(compare);

	return {
		queries: selected.queries.map((query) => queryById.get(query.queryId) as BenchmarkQuery),
		qrels: selected.qrels,
		corpus: selectedDocumentIds.map((documentId) => documentById.get(documentId) as CorpusDocument),
	};
}

class BoundedDistractorHeap {
	readonly #capacity: number;
	readonly #items: RankedDocument[] = [];

	constructor(capacity: number) {
		this.#capacity = capacity;
	}

	add(item: RankedDocument): void {
		if (this.#capacity === 0) {
			return;
		}
		if (this.#items.length < this.#capacity) {
			this.#items.push(item);
			this.#bubbleUp(this.#items.length - 1);
			return;
		}
		if (compareRanked(item, this.#items[0]) >= 0) {
			return;
		}
		this.#items[0] = item;
		this.#sinkDown(0);
	}

	values(): CorpusDocument[] {
		return [...this.#items].sort(compareRanked).map((item) => item.document);
	}

	#bubbleUp(startIndex: number): void {
		let index = startIndex;
		while (index > 0) {
			const parent = Math.floor((index - 1) / 2);
			if (compareRanked(this.#items[parent], this.#items[index]) >= 0) {
				break;
			}
			[this.#items[parent], this.#items[index]] = [this.#items[index], this.#items[parent]];
			index = parent;
		}
	}

	#sinkDown(startIndex: number): void {
		let index = startIndex;
		while (true) {
			const left = index * 2 + 1;
			const right = left + 1;
			let largest = index;
			if (left < this.#items.length && compareRanked(this.#items[left], this.#items[largest]) > 0) {
				largest = left;
			}
			if (right < this.#items.length && compareRanked(this.#items[right], this.#items[largest]) > 0) {
				largest = right;
			}
			if (largest === index) {
				return;
			}
			[this.#items[index], this.#items[largest]] = [this.#items[largest], this.#items[index]];
			index = largest;
		}
	}
}

class DiskBackedDuplicateDetector {
	readonly #directoryPath: string;
	readonly #directoryIdentity: PreparedDirectoryIdentity;
	readonly #counts = new Uint32Array(DUPLICATE_PARTITION_COUNT);
	readonly #bytes = new Float64Array(DUPLICATE_PARTITION_COUNT);
	readonly #openHandles = new Map<number, FileHandle>();
	readonly #fileIdentities = new Map<number, PreparedFileIdentity>();
	readonly #pending = Array<string>(DUPLICATE_PARTITION_COUNT).fill("");
	readonly #pendingBytes = new Uint32Array(DUPLICATE_PARTITION_COUNT);

	constructor(directoryPath: string, directoryIdentity: PreparedDirectoryIdentity) {
		this.#directoryPath = directoryPath;
		this.#directoryIdentity = directoryIdentity;
	}

	async record(documentId: string): Promise<void> {
		const partition = createHash("sha256").update(documentId, "utf8").digest()[0];
		const line = `${JSON.stringify(documentId)}\n`;
		const lineBytes = Buffer.byteLength(line, "utf8");
		if (this.#counts[partition] >= MAX_DUPLICATE_PARTITION_IDS) {
			throw new Error(`duplicate-check partition ${partition} exceeds ${MAX_DUPLICATE_PARTITION_IDS} document IDs`);
		}
		if (this.#bytes[partition] + lineBytes > MAX_DUPLICATE_PARTITION_BYTES) {
			throw new Error(`duplicate-check partition ${partition} exceeds ${MAX_DUPLICATE_PARTITION_BYTES} bytes`);
		}
		this.#pending[partition] += line;
		this.#pendingBytes[partition] += lineBytes;
		this.#counts[partition] += 1;
		this.#bytes[partition] += lineBytes;
		if (this.#pendingBytes[partition] >= DUPLICATE_WRITE_BUFFER_BYTES) {
			await this.#flushPartition(partition);
		}
	}

	async assertNoDuplicates(): Promise<void> {
		await this.#flushAll();
		await this.#closeHandles();
		for (let partition = 0; partition < DUPLICATE_PARTITION_COUNT; partition += 1) {
			const expectedCount = this.#counts[partition];
			if (expectedCount === 0) {
				continue;
			}
			const path = this.#pathFor(partition);
			await this.#assertOwnedPartition(partition);
			const input = createReadStream(path, { encoding: "utf8" });
			const reader = createInterface({ input, crlfDelay: Number.POSITIVE_INFINITY });
			const seen = new Set<string>();
			let count = 0;
			try {
				for await (const line of reader) {
					count += 1;
					const documentId = JSON.parse(line) as unknown;
					if (typeof documentId !== "string") {
						throw new Error(`invalid duplicate-check record in partition ${partition}`);
					}
					if (seen.has(documentId)) {
						throw new Error(`duplicate corpus document id: ${documentId}`);
					}
					seen.add(documentId);
				}
			} finally {
				reader.close();
				input.destroy();
			}
			if (count !== expectedCount) {
				throw new Error(
					`duplicate-check partition ${partition} expected ${expectedCount} records but read ${count}`,
				);
			}
			await this.#removeOwnedPartition(partition);
		}
	}

	async dispose(): Promise<void> {
		this.#pending.fill("");
		this.#pendingBytes.fill(0);
		try {
			await this.#closeHandles();
		} finally {
			for (const partition of [...this.#fileIdentities.keys()]) {
				await this.#removeOwnedPartition(partition).catch(() => undefined);
			}
			try {
				const current = await snapshotPreparedDirectory(this.#directoryPath);
				if (current.device === this.#directoryIdentity.device && current.inode === this.#directoryIdentity.inode) {
					await rmdir(this.#directoryPath);
				}
			} catch {
				// Non-empty or replaced duplicate-check directories are preserved.
			}
		}
	}

	async #getHandle(partition: number): Promise<FileHandle> {
		const existing = this.#openHandles.get(partition);
		if (existing !== undefined) {
			this.#openHandles.delete(partition);
			this.#openHandles.set(partition, existing);
			return existing;
		}
		if (this.#openHandles.size >= MAX_OPEN_DUPLICATE_PARTITIONS) {
			const oldest = this.#openHandles.entries().next().value as [number, FileHandle] | undefined;
			if (oldest !== undefined) {
				await oldest[1].close();
				this.#openHandles.delete(oldest[0]);
			}
		}
		const known = this.#fileIdentities.get(partition);
		const handle =
			known === undefined ? await this.#createPartition(partition) : await this.#reopenPartition(partition, known);
		this.#openHandles.set(partition, handle);
		return handle;
	}

	async #createPartition(partition: number): Promise<FileHandle> {
		const path = this.#pathFor(partition);
		const handle = await open(path, "wx", 0o600);
		try {
			const stats = await handle.stat();
			const pathStats = await lstat(path);
			const mode = Number(stats.mode) & 0o777;
			if (
				!stats.isFile() ||
				(mode & 0o077) !== 0 ||
				!pathStats.isFile() ||
				pathStats.isSymbolicLink() ||
				pathStats.dev !== stats.dev ||
				pathStats.ino !== stats.ino ||
				(Number(pathStats.mode) & 0o777) !== mode
			) {
				throw new Error(`duplicate-check partition ${partition} is not an exact private regular file`);
			}
			this.#fileIdentities.set(partition, { device: stats.dev, inode: stats.ino, mode });
			return handle;
		} catch (error) {
			await handle.close();
			throw error;
		}
	}

	async #reopenPartition(partition: number, identity: PreparedFileIdentity): Promise<FileHandle> {
		const path = this.#pathFor(partition);
		try {
			this.#assertPartitionStats(await lstat(path), partition, identity);
		} catch {
			throw new Error(`duplicate-check partition ${partition} changed`);
		}
		const noFollow = process.platform === "win32" ? 0 : constants.O_NOFOLLOW;
		let handle: FileHandle | undefined;
		try {
			handle = await open(path, constants.O_WRONLY | constants.O_APPEND | noFollow);
			this.#assertPartitionStats(await handle.stat(), partition, identity);
			this.#assertPartitionStats(await lstat(path), partition, identity);
			return handle;
		} catch {
			await handle?.close().catch(() => undefined);
			throw new Error(`duplicate-check partition ${partition} changed`);
		}
	}

	#assertPartitionStats(
		stats: Awaited<ReturnType<typeof lstat>>,
		partition: number,
		identity: PreparedFileIdentity,
	): void {
		if (
			!stats.isFile() ||
			stats.isSymbolicLink() ||
			stats.dev !== identity.device ||
			stats.ino !== identity.inode ||
			(Number(stats.mode) & 0o777) !== identity.mode ||
			(Number(stats.mode) & 0o077) !== 0
		) {
			throw new Error(`duplicate-check partition ${partition} changed`);
		}
	}

	async #flushAll(): Promise<void> {
		for (let partition = 0; partition < DUPLICATE_PARTITION_COUNT; partition += 1) {
			await this.#flushPartition(partition);
		}
	}

	async #flushPartition(partition: number): Promise<void> {
		const pending = this.#pending[partition];
		if (pending.length === 0) {
			return;
		}
		const handle = await this.#getHandle(partition);
		await handle.writeFile(pending, "utf8");
		this.#pending[partition] = "";
		this.#pendingBytes[partition] = 0;
	}

	async #closeHandles(): Promise<void> {
		let firstError: unknown;
		for (const [partition, handle] of [...this.#openHandles]) {
			try {
				await handle.close();
				this.#openHandles.delete(partition);
			} catch (error) {
				firstError ??= error;
			}
		}
		if (firstError !== undefined) {
			throw firstError;
		}
	}

	#pathFor(partition: number): string {
		return join(this.#directoryPath, `partition-${partition.toString(16).padStart(2, "0")}.jsonl`);
	}

	async #assertOwnedPartition(partition: number): Promise<void> {
		const identity = this.#fileIdentities.get(partition);
		if (identity === undefined) throw new Error(`duplicate-check partition ${partition} has no identity`);
		this.#assertPartitionStats(await lstat(this.#pathFor(partition)), partition, identity);
	}

	async #removeOwnedPartition(partition: number): Promise<void> {
		await this.#assertOwnedPartition(partition);
		await unlink(this.#pathFor(partition));
		this.#fileIdentities.delete(partition);
	}
}

class DecompressedLimitTransform extends Transform {
	#totalBytes = 0;
	#lineBytes = 0;
	#lineNumber = 1;
	readonly #path: string;
	readonly #maxTotalBytes: number;

	constructor(path: string, maxTotalBytes: number) {
		super();
		this.#path = path;
		this.#maxTotalBytes = maxTotalBytes;
	}

	_transform(chunk: Buffer, encoding: BufferEncoding, callback: TransformCallback): void {
		const bytes = typeof chunk === "string" ? Buffer.from(chunk, encoding) : chunk;
		this.#totalBytes += bytes.byteLength;
		if (this.#totalBytes > this.#maxTotalBytes) {
			callback(new Error(`decompressed data in ${this.#path} exceeds the configured byte limit`));
			return;
		}
		for (const byte of bytes) {
			if (byte === 0x0a) {
				this.#lineBytes = 0;
				this.#lineNumber += 1;
				continue;
			}
			this.#lineBytes += 1;
			if (this.#lineBytes > MAX_CORPUS_LINE_BYTES) {
				callback(new Error(`line ${this.#lineNumber} in ${this.#path} exceeds 16 MiB`));
				return;
			}
		}
		callback(null, chunk);
	}
}

function parseCorpusDocument(line: string, path: string, lineNumber: number): CorpusDocument {
	let value: unknown;
	try {
		value = JSON.parse(line);
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		throw new Error(`invalid corpus JSON at ${path}:${lineNumber}: ${message}`);
	}
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`corpus record at ${path}:${lineNumber} must be an object`);
	}
	const record = value as Record<string, unknown>;
	if (typeof record.docid !== "string" || record.docid.trim().length === 0) {
		throw new Error(`corpus docid at ${path}:${lineNumber} must not be blank`);
	}
	if (typeof record.title !== "string" || typeof record.text !== "string") {
		throw new Error(`corpus title and text at ${path}:${lineNumber} must be strings`);
	}
	return { documentId: record.docid, title: record.title, text: record.text };
}

async function forEachGzipCorpusDocument(
	path: string,
	maxDecompressedBytes: number,
	visit: (document: CorpusDocument) => void | Promise<void>,
): Promise<void> {
	const input = createReadStream(path);
	const gunzip = createGunzip();
	const limiter = new DecompressedLimitTransform(path, maxDecompressedBytes);
	const reader = createInterface({
		input: input.pipe(gunzip).pipe(limiter),
		crlfDelay: Number.POSITIVE_INFINITY,
	});
	let lineNumber = 0;
	try {
		for await (const line of reader) {
			lineNumber += 1;
			if (line.trim().length === 0) {
				throw new Error(`corpus line at ${path}:${lineNumber} must not be blank`);
			}
			await visit(parseCorpusDocument(line, path, lineNumber));
		}
	} finally {
		reader.close();
		input.destroy();
		gunzip.destroy();
		limiter.destroy();
	}
	if (lineNumber === 0) {
		throw new Error(`corpus shard ${path} is empty`);
	}
}

function parseContentLength(response: Response, maxBytes: number, url: string): number | undefined {
	const value = response.headers.get("content-length");
	if (value === null) {
		return undefined;
	}
	if (!/^\d+$/.test(value)) {
		throw new Error(`invalid content-length from ${url}`);
	}
	const bytes = Number(value);
	if (!Number.isSafeInteger(bytes)) {
		throw new Error(`content-length from ${url} is not a safe integer`);
	}
	if (bytes > maxBytes) {
		throw new Error(`download from ${url} exceeds the configured byte limit`);
	}
	return bytes;
}

function isAllowedDownloadHost(hostname: string): boolean {
	const normalized = hostname.toLowerCase();
	return normalized === "huggingface.co" || normalized.endsWith(".huggingface.co") || normalized.endsWith(".hf.co");
}

async function writeBytes(handle: FileHandle, bytes: Uint8Array, position: number): Promise<number> {
	let offset = 0;
	while (offset < bytes.byteLength) {
		const result = await handle.write(bytes, offset, bytes.byteLength - offset, position + offset);
		if (result.bytesWritten === 0) {
			throw new Error("file write made no progress");
		}
		offset += result.bytesWritten;
	}
	return position + offset;
}

async function downloadPinnedFile(
	url: string,
	destinationPath: string,
	relativePath: string,
	fetchImpl: FetchLike,
	maxRedirects: number,
	maxBytes: number,
	timeoutMs: number,
): Promise<DownloadResult> {
	const controller = new AbortController();
	const timeoutError = new Error(`download from ${url} timed out after ${timeoutMs} ms`);
	let rejectTimeout!: (reason: Error) => void;
	const timeout = new Promise<never>((_resolve, reject) => {
		rejectTimeout = reject;
	});
	const timer = setTimeout(() => {
		controller.abort(timeoutError);
		rejectTimeout(timeoutError);
	}, timeoutMs);
	let currentUrl = new URL(url);
	let response: Response | undefined;
	try {
		for (let redirectCount = 0; ; redirectCount += 1) {
			response = await Promise.race([
				fetchImpl(currentUrl, { method: "GET", redirect: "manual", signal: controller.signal }),
				timeout,
			]);
			if (!REDIRECT_STATUSES.has(response.status)) {
				break;
			}
			if (redirectCount >= maxRedirects) {
				await response.body?.cancel();
				throw new Error(`download from ${url} exceeded ${maxRedirects} redirects`);
			}
			const location = response.headers.get("location");
			if (location === null) {
				await response.body?.cancel();
				throw new Error(`redirect from ${currentUrl.toString()} has no location`);
			}
			const nextUrl = new URL(location, currentUrl);
			if (nextUrl.protocol !== "https:") {
				await response.body?.cancel();
				throw new Error(`redirect from ${currentUrl.toString()} must use HTTPS`);
			}
			if (!isAllowedDownloadHost(nextUrl.hostname)) {
				await response.body?.cancel();
				throw new Error(`redirect from ${currentUrl.toString()} is outside the allowed download hosts`);
			}
			await response.body?.cancel();
			currentUrl = nextUrl;
		}
		if (!response.ok) {
			await response.body?.cancel();
			throw new Error(`download from ${currentUrl.toString()} failed with HTTP ${response.status}`);
		}
		if (response.body === null) {
			throw new Error(`download from ${currentUrl.toString()} returned no body`);
		}
		const body = response.body;
		let expectedBytes: number | undefined;
		let handle: FileHandle;
		try {
			expectedBytes = parseContentLength(response, maxBytes, currentUrl.toString());
			handle = await open(destinationPath, "wx", 0o600);
		} catch (error) {
			await body.cancel().catch(() => undefined);
			throw error;
		}
		const hash = createHash("sha256");
		let bytes = 0;
		const reader = body.getReader();
		try {
			while (true) {
				const { done, value } = await Promise.race([reader.read(), timeout]);
				if (done) {
					break;
				}
				if (bytes + value.byteLength > maxBytes) {
					await reader.cancel();
					throw new Error(`download from ${currentUrl.toString()} exceeds the configured byte limit`);
				}
				hash.update(value);
				bytes = await writeBytes(handle, value, bytes);
			}
			if (expectedBytes !== undefined && expectedBytes !== bytes) {
				throw new Error(
					`download from ${currentUrl.toString()} declared ${expectedBytes} bytes but returned ${bytes}`,
				);
			}
		} catch (error) {
			await reader.cancel().catch(() => undefined);
			throw error;
		} finally {
			reader.releaseLock();
			await handle.close();
		}
		return { url, path: relativePath, sha256: hash.digest("hex"), bytes };
	} finally {
		clearTimeout(timer);
	}
}

async function writeJsonLinesFile(path: string, records: readonly unknown[]): Promise<NormalizedPreparedFile> {
	const handle = await open(path, "wx", 0o600);
	const hash = createHash("sha256");
	let position = 0;
	try {
		for (const record of records) {
			const serialized = JSON.stringify(record);
			if (serialized === undefined) {
				throw new Error(`record for ${path} is not JSON serializable`);
			}
			const bytes = Buffer.from(`${serialized}\n`, "utf8");
			hash.update(bytes);
			position = await writeBytes(handle, bytes, position);
		}
		await handle.sync();
	} finally {
		await handle.close();
	}
	return { sha256: hash.digest("hex"), bytes: position, records: records.length };
}

interface JsonLinesWriter {
	readonly path: string;
	readonly handle: FileHandle;
	readonly hash: ReturnType<typeof createHash>;
	position: number;
	records: number;
	closed: boolean;
}

async function openJsonLinesWriter(path: string): Promise<JsonLinesWriter> {
	return {
		path,
		handle: await open(path, "wx", 0o600),
		hash: createHash("sha256"),
		position: 0,
		records: 0,
		closed: false,
	};
}

async function appendJsonLine(writer: JsonLinesWriter, record: unknown): Promise<void> {
	if (writer.closed) {
		throw new Error(`normalized output ${writer.path} is closed`);
	}
	const serialized = JSON.stringify(record);
	if (serialized === undefined) {
		throw new Error(`record for ${writer.path} is not JSON serializable`);
	}
	const bytes = Buffer.from(`${serialized}\n`, "utf8");
	writer.hash.update(bytes);
	writer.position = await writeBytes(writer.handle, bytes, writer.position);
	writer.records += 1;
}

async function finalizeJsonLinesWriter(writer: JsonLinesWriter): Promise<NormalizedPreparedFile> {
	if (writer.closed) {
		throw new Error(`normalized output ${writer.path} is closed`);
	}
	writer.closed = true;
	try {
		await writer.handle.sync();
	} finally {
		await writer.handle.close();
	}
	return {
		sha256: writer.hash.digest("hex"),
		bytes: writer.position,
		records: writer.records,
	};
}

async function closeJsonLinesWriter(writer: JsonLinesWriter): Promise<void> {
	if (writer.closed) return;
	writer.closed = true;
	await writer.handle.close();
}

function assertManifest(condition: unknown, message: string): asserts condition {
	if (!condition) {
		throw new Error(`invalid prepared manifest: ${message}`);
	}
}

function validateStringIds(value: unknown, label: string): string[] {
	assertManifest(Array.isArray(value), `${label} must be an array`);
	const ids = value as unknown[];
	assertManifest(
		ids.every((id) => typeof id === "string" && id.trim().length > 0),
		`${label} must contain non-blank strings`,
	);
	assertManifest(new Set(ids).size === ids.length, `${label} must not contain duplicates`);
	return [...(ids as string[])];
}

function exactManifestRecord(value: unknown, keys: readonly string[], label: string): Record<string, unknown> {
	assertManifest(typeof value === "object" && value !== null && !Array.isArray(value), `${label} must be an object`);
	const record = value as Record<string, unknown>;
	const allowed = new Set(keys);
	for (const key of Object.keys(record)) {
		assertManifest(allowed.has(key), `${label} has unknown field ${key}`);
	}
	for (const key of keys) {
		assertManifest(key in record, `${label} is missing field ${key}`);
	}
	return record;
}

function validateSource(value: unknown, expectedUrl: string, expectedPath: string, label: string): PreparedSource {
	const source = exactManifestRecord(value, ["url", "path", "sha256", "bytes"], label);
	assertManifest(source.url === expectedUrl, `${label}.url does not match the pinned source`);
	assertManifest(source.path === expectedPath, `${label}.path must be ${expectedPath}`);
	assertManifest(
		typeof source.sha256 === "string" && SHA256_PATTERN.test(source.sha256),
		`${label}.sha256 is invalid`,
	);
	assertManifest(Number.isSafeInteger(source.bytes) && (source.bytes as number) > 0, `${label}.bytes is invalid`);
	return {
		url: source.url,
		path: source.path,
		sha256: source.sha256,
		bytes: source.bytes as number,
	} as PreparedSource;
}

function validateCommonPreparedManifest(manifest: Record<string, unknown>): Omit<PreparedManifestBase, "normalized"> {
	assertManifest(manifest.schemaVersion === 1, "schemaVersion must be 1");
	assertManifest(
		manifest.normalizationVersion === MIRACL_NORMALIZATION_VERSION,
		`normalizationVersion must be ${MIRACL_NORMALIZATION_VERSION}`,
	);
	assertManifest(manifest.profile === "smoke" || manifest.profile === "full", "profile must be smoke or full");

	const revisions = exactManifestRecord(manifest.revisions, ["topics", "corpus"], "revisions");
	assertManifest(revisions.topics === MIRACL_SOURCES.topics.revision, "topics revision does not match the pin");
	assertManifest(revisions.corpus === MIRACL_SOURCES.corpus.revision, "corpus revision does not match the pin");

	const sources = exactManifestRecord(manifest.sources, ["topics", "qrels", "corpus"], "sources");
	const topics = validateSource(
		sources.topics,
		MIRACL_SOURCES.topics.topicsUrl,
		"downloads/topics.tsv",
		"sources.topics",
	);
	const qrels = validateSource(sources.qrels, MIRACL_SOURCES.topics.qrelsUrl, "downloads/qrels.tsv", "sources.qrels");
	assertManifest(Array.isArray(sources.corpus), "sources.corpus must be an array");
	assertManifest(
		sources.corpus.length === MIRACL_SOURCES.corpus.urls.length,
		"sources.corpus must contain three shards",
	);
	const corpus: PreparedSource[] = [];
	for (const [index, url] of MIRACL_SOURCES.corpus.urls.entries()) {
		corpus.push(
			validateSource(sources.corpus[index], url, `downloads/docs-${index}.jsonl.gz`, `sources.corpus[${index}]`),
		);
	}

	const files = exactManifestRecord(manifest.files, ["queries", "qrels", "corpus"], "files");
	assertManifest(files.queries === "queries.jsonl", "files.queries is invalid");
	assertManifest(files.qrels === "qrels.jsonl", "files.qrels is invalid");
	assertManifest(files.corpus === "corpus.jsonl", "files.corpus is invalid");
	return {
		schemaVersion: 1,
		normalizationVersion: MIRACL_NORMALIZATION_VERSION,
		profile: manifest.profile,
		revisions: {
			topics: revisions.topics,
			corpus: revisions.corpus,
		} as PreparedManifestBase["revisions"],
		sources: { topics, qrels, corpus },
		files: {
			queries: "queries.jsonl",
			qrels: "qrels.jsonl",
			corpus: "corpus.jsonl",
		},
	};
}

function validateNormalizedFile(value: unknown, expectedRecords: number, label: string): NormalizedPreparedFile {
	const file = exactManifestRecord(value, ["sha256", "bytes", "records"], label);
	assertManifest(typeof file.sha256 === "string" && SHA256_PATTERN.test(file.sha256), `${label}.sha256 is invalid`);
	assertManifest(Number.isSafeInteger(file.bytes) && (file.bytes as number) > 0, `${label}.bytes is invalid`);
	assertManifest(file.records === expectedRecords, `${label}.records does not match counts`);
	return {
		sha256: file.sha256,
		bytes: file.bytes as number,
		records: file.records,
	} as NormalizedPreparedFile;
}

export function validatePreparedManifest(value: unknown, options: PrepareValidationOptions = {}): PreparedManifest {
	assertManifest(typeof value === "object" && value !== null && !Array.isArray(value), "root must be an object");
	const manifest = value as Record<string, unknown>;
	assertManifest(manifest.profile === "smoke" || manifest.profile === "full", "profile must be smoke or full");
	if (manifest.profile === "full") {
		exactManifestRecord(
			manifest,
			["schemaVersion", "normalizationVersion", "profile", "revisions", "sources", "counts", "normalized", "files"],
			"root",
		);
		const common = validateCommonPreparedManifest(manifest);
		const counts = exactManifestRecord(
			manifest.counts,
			["queries", "qrels", "positiveQrels", "corpus", "judgedDocuments"],
			"counts",
		);
		for (const field of ["queries", "qrels", "positiveQrels", "corpus", "judgedDocuments"]) {
			assertManifest(
				Number.isSafeInteger(counts[field]) && (counts[field] as number) >= 0,
				`counts.${field} must be a non-negative safe integer`,
			);
		}
		const expectedCorpusPassages = options.expectedFullCorpusPassages ?? MIRACL_FULL_CORPUS_PASSAGES;
		assertCount(expectedCorpusPassages, "expectedFullCorpusPassages", false);
		assertManifest(counts.corpus === expectedCorpusPassages, `full corpus count must be ${expectedCorpusPassages}`);
		const queryCount = counts.queries as number;
		const qrelCount = counts.qrels as number;
		const positiveQrelCount = counts.positiveQrels as number;
		const judgedDocumentCount = counts.judgedDocuments as number;
		assertManifest(queryCount > 0, "full manifest must contain at least one query");
		assertManifest(qrelCount > 0, "full manifest must contain at least one qrel");
		assertManifest(qrelCount >= positiveQrelCount, "qrel count is lower than positive qrel count");
		assertManifest(judgedDocumentCount > 0, "full manifest must contain at least one judged document");
		assertManifest(judgedDocumentCount <= qrelCount, "judged document count exceeds qrels");
		assertManifest(judgedDocumentCount <= expectedCorpusPassages, "judged document count exceeds corpus");
		assertManifest(
			typeof manifest.normalized === "object" && manifest.normalized !== null,
			"normalized must be an object",
		);
		const normalized = exactManifestRecord(manifest.normalized, ["queries", "qrels", "corpus"], "normalized");
		return {
			...common,
			profile: "full",
			counts: {
				queries: queryCount,
				qrels: qrelCount,
				positiveQrels: positiveQrelCount,
				corpus: expectedCorpusPassages,
				judgedDocuments: judgedDocumentCount,
			},
			normalized: {
				queries: validateNormalizedFile(normalized.queries, queryCount, "normalized.queries"),
				qrels: validateNormalizedFile(normalized.qrels, qrelCount, "normalized.qrels"),
				corpus: validateNormalizedFile(normalized.corpus, expectedCorpusPassages, "normalized.corpus"),
			},
		};
	}

	exactManifestRecord(
		manifest,
		[
			"schemaVersion",
			"normalizationVersion",
			"profile",
			"revisions",
			"sources",
			"seed",
			"selectedIds",
			"counts",
			"normalized",
			"files",
		],
		"root",
	);
	const common = validateCommonPreparedManifest(manifest);
	assertManifest(Number.isSafeInteger(manifest.seed), "seed must be a safe integer");
	const seed = manifest.seed as number;
	const selectedIds = exactManifestRecord(manifest.selectedIds, ["queryIds", "documentIds"], "selectedIds");
	const queryIds = validateStringIds(selectedIds.queryIds, "selectedIds.queryIds");
	const documentIds = validateStringIds(selectedIds.documentIds, "selectedIds.documentIds");
	const compare = compareIds(seed);
	const sortedQueryIds = [...queryIds].sort(compare);
	const sortedDocumentIds = [...documentIds].sort(compare);
	assertManifest(
		queryIds.every((queryId, index) => queryId === sortedQueryIds[index]),
		"selectedIds.queryIds are not in deterministic order",
	);
	assertManifest(
		documentIds.every((documentId, index) => documentId === sortedDocumentIds[index]),
		"selectedIds.documentIds are not in deterministic order",
	);

	const counts = exactManifestRecord(
		manifest.counts,
		["queries", "qrels", "positiveQrels", "corpus", "judgedDocuments", "distractors"],
		"counts",
	);
	for (const field of ["queries", "qrels", "positiveQrels", "corpus", "judgedDocuments", "distractors"]) {
		assertManifest(
			Number.isSafeInteger(counts[field]) && (counts[field] as number) >= 0,
			`counts.${field} must be a non-negative safe integer`,
		);
	}
	assertManifest(counts.queries === queryIds.length, "query count does not match selected query IDs");
	assertManifest(counts.corpus === documentIds.length, "corpus count does not match selected document IDs");
	const queryCount = counts.queries as number;
	const qrelCount = counts.qrels as number;
	const positiveQrelCount = counts.positiveQrels as number;
	const corpusCount = counts.corpus as number;
	const judgedDocumentCount = counts.judgedDocuments as number;
	const distractorCount = counts.distractors as number;
	assertManifest(queryCount > 0, "smoke manifest must contain at least one selected query");
	assertManifest(qrelCount >= positiveQrelCount, "qrel count is lower than positive qrel count");
	assertManifest(positiveQrelCount >= queryCount, "positive qrel count must cover every selected query");
	assertManifest(judgedDocumentCount > 0, "smoke manifest must contain at least one judged document");
	assertManifest(judgedDocumentCount <= qrelCount, "judged document count exceeds qrels");
	assertManifest(judgedDocumentCount <= corpusCount, "judged document count exceeds corpus");
	assertManifest(
		BigInt(qrelCount) <= BigInt(queryCount) * BigInt(judgedDocumentCount),
		"qrel count exceeds the possible query/document pairs",
	);
	assertManifest(
		distractorCount === corpusCount - judgedDocumentCount,
		"distractor count does not match corpus minus judged documents",
	);
	const normalized = exactManifestRecord(manifest.normalized, ["queries", "qrels", "corpus"], "normalized");

	return {
		...common,
		profile: "smoke",
		seed,
		selectedIds: { queryIds, documentIds },
		counts: {
			queries: queryCount,
			qrels: qrelCount,
			positiveQrels: positiveQrelCount,
			corpus: corpusCount,
			judgedDocuments: judgedDocumentCount,
			distractors: distractorCount,
		},
		normalized: {
			queries: validateNormalizedFile(normalized.queries, queryCount, "normalized.queries"),
			qrels: validateNormalizedFile(normalized.qrels, qrelCount, "normalized.qrels"),
			corpus: validateNormalizedFile(normalized.corpus, corpusCount, "normalized.corpus"),
		},
	};
}

export function prepareMiracl(
	options: FullPrepareOptions,
	validationOptions?: PrepareValidationOptions,
): Promise<FullPreparedManifest>;
export function prepareMiracl(
	options: SmokePrepareOptions,
	validationOptions?: PrepareValidationOptions,
): Promise<SmokePreparedManifest>;
export async function prepareMiracl(
	options: PrepareOptions,
	validationOptions: PrepareValidationOptions = {},
): Promise<PreparedManifest> {
	assertNonBlank(options.outputDir, "outputDir");
	const profile = options.profile ?? "smoke";
	const smokeOptions = options as SmokePrepareOptions;
	const selectionOptions: SmokeSelectionOptions | undefined =
		profile === "smoke"
			? {
					seed: smokeOptions.seed ?? MIRACL_SMOKE_PROFILE.seed,
					queryCount: smokeOptions.queryCount ?? MIRACL_SMOKE_PROFILE.queryCount,
					distractorCount: smokeOptions.distractorCount ?? MIRACL_SMOKE_PROFILE.distractorCount,
				}
			: undefined;
	if (selectionOptions !== undefined) validateSelectionOptions(selectionOptions);
	const expectedFullCorpusPassages = validationOptions.expectedFullCorpusPassages ?? MIRACL_FULL_CORPUS_PASSAGES;
	assertCount(expectedFullCorpusPassages, "expectedFullCorpusPassages", false);
	const maxRedirects = options.maxRedirects ?? DEFAULT_MAX_REDIRECTS;
	const maxDownloadBytes = options.maxDownloadBytes ?? DEFAULT_MAX_DOWNLOAD_BYTES;
	const maxDecompressedBytes = options.maxDecompressedBytesPerShard ?? DEFAULT_MAX_DECOMPRESSED_BYTES_PER_SHARD;
	const fetchTimeoutMs = options.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS;
	assertCount(maxRedirects, "maxRedirects", true);
	assertCount(maxDownloadBytes, "maxDownloadBytes", false);
	assertCount(maxDecompressedBytes, "maxDecompressedBytesPerShard", false);
	assertCount(fetchTimeoutMs, "fetchTimeoutMs", false);
	const fetchImpl = options.fetchImpl ?? fetch;

	let outputIdentity: PreparedDirectoryIdentity | undefined;
	try {
		await mkdir(options.outputDir, { mode: 0o700 });
		outputIdentity = await snapshotPreparedDirectory(options.outputDir);
		const downloadsDir = join(options.outputDir, "downloads");
		await mkdir(downloadsDir, { mode: 0o700 });
		const topics = await downloadPinnedFile(
			MIRACL_SOURCES.topics.topicsUrl,
			join(downloadsDir, "topics.tsv"),
			"downloads/topics.tsv",
			fetchImpl,
			maxRedirects,
			maxDownloadBytes,
			fetchTimeoutMs,
		);
		const qrelsSource = await downloadPinnedFile(
			MIRACL_SOURCES.topics.qrelsUrl,
			join(downloadsDir, "qrels.tsv"),
			"downloads/qrels.tsv",
			fetchImpl,
			maxRedirects,
			maxDownloadBytes,
			fetchTimeoutMs,
		);
		const corpusSources: PreparedSource[] = [];
		for (const [index, url] of MIRACL_SOURCES.corpus.urls.entries()) {
			corpusSources.push(
				await downloadPinnedFile(
					url,
					join(downloadsDir, `docs-${index}.jsonl.gz`),
					`downloads/docs-${index}.jsonl.gz`,
					fetchImpl,
					maxRedirects,
					maxDownloadBytes,
					fetchTimeoutMs,
				),
			);
		}

		const allQueries = await readTopicsTsv(join(downloadsDir, "topics.tsv"), {
			totalBytes: topics.bytes,
			maxRecords: MAX_TOPIC_RECORDS,
			requirePrivateFile: true,
			label: "downloaded topics",
		});
		const allQrels = await readQrels(join(downloadsDir, "qrels.tsv"), {
			totalBytes: qrelsSource.bytes,
			maxRecords: MAX_QREL_RECORDS,
			requirePrivateFile: true,
			label: "downloaded qrels",
		});
		validateQueriesAndQrels(allQueries, allQrels);
		if (profile === "full") {
			const queriesNormalized = await writeJsonLinesFile(join(options.outputDir, "queries.jsonl"), allQueries);
			const qrelsNormalized = await writeJsonLinesFile(join(options.outputDir, "qrels.jsonl"), allQrels);
			const allReferencedIds = new Set(allQrels.map((qrel) => qrel.documentId));
			const seenReferencedIds = new Set<string>();
			const duplicateCheckDirectory = join(options.outputDir, ".duplicate-check");
			await mkdir(duplicateCheckDirectory, { mode: 0o700 });
			const duplicateDetector = new DiskBackedDuplicateDetector(
				duplicateCheckDirectory,
				await snapshotPreparedDirectory(duplicateCheckDirectory),
			);
			const corpusWriter = await openJsonLinesWriter(join(options.outputDir, "corpus.jsonl"));
			let corpusNormalized: NormalizedPreparedFile;
			try {
				for (const [index] of MIRACL_SOURCES.corpus.urls.entries()) {
					const path = join(downloadsDir, `docs-${index}.jsonl.gz`);
					await forEachGzipCorpusDocument(path, maxDecompressedBytes, async (document) => {
						await duplicateDetector.record(document.documentId);
						if (allReferencedIds.has(document.documentId)) {
							seenReferencedIds.add(document.documentId);
						}
						await appendJsonLine(corpusWriter, document);
					});
				}
				await duplicateDetector.assertNoDuplicates();
				corpusNormalized = await finalizeJsonLinesWriter(corpusWriter);
			} catch (error) {
				await closeJsonLinesWriter(corpusWriter).catch(() => undefined);
				throw error;
			} finally {
				await duplicateDetector.dispose();
			}
			const missingReferencedIds = [...allReferencedIds].filter((documentId) => !seenReferencedIds.has(documentId));
			if (missingReferencedIds.length > 0) {
				throw new Error(`qrels reference missing corpus document: ${missingReferencedIds[0]}`);
			}
			if (corpusNormalized.records !== expectedFullCorpusPassages) {
				throw new Error(
					`full MIRACL Korean corpus must contain ${expectedFullCorpusPassages} passages but contained ${corpusNormalized.records}`,
				);
			}
			const manifest: FullPreparedManifest = {
				schemaVersion: 1,
				normalizationVersion: MIRACL_NORMALIZATION_VERSION,
				profile: "full",
				revisions: {
					topics: MIRACL_SOURCES.topics.revision,
					corpus: MIRACL_SOURCES.corpus.revision,
				},
				sources: {
					topics,
					qrels: qrelsSource,
					corpus: corpusSources,
				},
				counts: {
					queries: allQueries.length,
					qrels: allQrels.length,
					positiveQrels: allQrels.filter((qrel) => qrel.relevance > 0).length,
					corpus: corpusNormalized.records,
					judgedDocuments: allReferencedIds.size,
				},
				normalized: {
					queries: queriesNormalized,
					qrels: qrelsNormalized,
					corpus: corpusNormalized,
				},
				files: {
					queries: "queries.jsonl",
					qrels: "qrels.jsonl",
					corpus: "corpus.jsonl",
				},
			};
			validatePreparedManifest(manifest, { expectedFullCorpusPassages });
			await writeJsonAtomic(join(options.outputDir, "prepared-manifest.json"), manifest);
			return manifest;
		}
		const smokeSelectionOptions = selectionOptions as SmokeSelectionOptions;
		const selected = selectQueriesAndQrels(allQueries, allQrels, smokeSelectionOptions);
		const selectedJudgedIds = new Set(selected.qrels.map((qrel) => qrel.documentId));
		const allReferencedIds = new Set(allQrels.map((qrel) => qrel.documentId));
		const seenReferencedIds = new Set<string>();
		const judgedDocuments = new Map<string, CorpusDocument>();
		const distractors = new BoundedDistractorHeap(smokeSelectionOptions.distractorCount);
		const duplicateCheckDirectory = join(options.outputDir, ".duplicate-check");
		await mkdir(duplicateCheckDirectory, { mode: 0o700 });
		const duplicateDetector = new DiskBackedDuplicateDetector(
			duplicateCheckDirectory,
			await snapshotPreparedDirectory(duplicateCheckDirectory),
		);
		try {
			for (const [index] of MIRACL_SOURCES.corpus.urls.entries()) {
				const path = join(downloadsDir, `docs-${index}.jsonl.gz`);
				await forEachGzipCorpusDocument(path, maxDecompressedBytes, async (document) => {
					await duplicateDetector.record(document.documentId);
					if (allReferencedIds.has(document.documentId)) {
						seenReferencedIds.add(document.documentId);
					}
					if (selectedJudgedIds.has(document.documentId)) {
						judgedDocuments.set(document.documentId, document);
						return;
					}
					distractors.add({
						key: deterministicKey(smokeSelectionOptions.seed, document.documentId),
						document,
					});
				});
			}
			await duplicateDetector.assertNoDuplicates();
		} finally {
			await duplicateDetector.dispose();
		}
		const missingReferencedIds = [...allReferencedIds].filter((documentId) => !seenReferencedIds.has(documentId));
		if (missingReferencedIds.length > 0) {
			throw new Error(`qrels reference missing corpus document: ${missingReferencedIds[0]}`);
		}
		const selectedDistractors = distractors.values();
		if (selectedDistractors.length < smokeSelectionOptions.distractorCount) {
			throw new Error(
				`requested ${smokeSelectionOptions.distractorCount} distractors but only ${selectedDistractors.length} unjudged documents are available`,
			);
		}
		const compare = compareIds(smokeSelectionOptions.seed);
		const selectedCorpus = [...judgedDocuments.values(), ...selectedDistractors].sort((left, right) =>
			compare(left.documentId, right.documentId),
		);

		const queriesNormalized = await writeJsonLinesFile(join(options.outputDir, "queries.jsonl"), selected.queries);
		const qrelsNormalized = await writeJsonLinesFile(join(options.outputDir, "qrels.jsonl"), selected.qrels);
		const corpusNormalized = await writeJsonLinesFile(join(options.outputDir, "corpus.jsonl"), selectedCorpus);
		const manifest: SmokePreparedManifest = {
			schemaVersion: 1,
			normalizationVersion: MIRACL_NORMALIZATION_VERSION,
			profile: "smoke",
			revisions: {
				topics: MIRACL_SOURCES.topics.revision,
				corpus: MIRACL_SOURCES.corpus.revision,
			},
			sources: {
				topics,
				qrels: qrelsSource,
				corpus: corpusSources,
			},
			seed: smokeSelectionOptions.seed,
			selectedIds: {
				queryIds: selected.queries.map((query) => query.queryId),
				documentIds: selectedCorpus.map((document) => document.documentId),
			},
			counts: {
				queries: selected.queries.length,
				qrels: selected.qrels.length,
				positiveQrels: selected.qrels.filter((qrel) => qrel.relevance > 0).length,
				corpus: selectedCorpus.length,
				judgedDocuments: judgedDocuments.size,
				distractors: selectedDistractors.length,
			},
			normalized: {
				queries: queriesNormalized,
				qrels: qrelsNormalized,
				corpus: corpusNormalized,
			},
			files: {
				queries: "queries.jsonl",
				qrels: "qrels.jsonl",
				corpus: "corpus.jsonl",
			},
		};
		validatePreparedManifest(manifest);
		await writeJsonAtomic(join(options.outputDir, "prepared-manifest.json"), manifest);
		return manifest;
	} catch (error) {
		if (outputIdentity !== undefined) {
			await cleanupOwnedPreparedDirectory(options.outputDir, outputIdentity).catch(() => undefined);
		}
		throw error;
	}
}

async function snapshotPreparedDirectory(path: string): Promise<PreparedDirectoryIdentity> {
	const stats = await lstat(path);
	if (!stats.isDirectory() || stats.isSymbolicLink()) {
		throw new Error("prepared output directory changed");
	}
	return { device: stats.dev, inode: stats.ino };
}

async function cleanupOwnedPreparedDirectory(path: string, identity: PreparedDirectoryIdentity): Promise<void> {
	let ownership: PreparedTreeOwnership;
	try {
		ownership = await snapshotPreparedTree(path, identity);
	} catch {
		return;
	}
	for (const [relativePath, childIdentity] of ownership.files) {
		const child = join(path, relativePath);
		try {
			const stats = await lstat(child);
			if (
				stats.isFile() &&
				!stats.isSymbolicLink() &&
				stats.dev === childIdentity.device &&
				stats.ino === childIdentity.inode
			) {
				await unlink(child);
			}
		} catch {
			// Replaced or missing files are preserved.
		}
	}
	const directories = [...ownership.directories].sort(
		([left], [right]) => right.split(/[\\/]/u).length - left.split(/[\\/]/u).length,
	);
	for (const [relativePath, childIdentity] of directories) {
		const child = join(path, relativePath);
		try {
			const stats = await lstat(child);
			if (
				stats.isDirectory() &&
				!stats.isSymbolicLink() &&
				stats.dev === childIdentity.device &&
				stats.ino === childIdentity.inode
			) {
				await rmdir(child);
			}
		} catch {
			// Non-empty, replaced, or missing directories are preserved.
		}
	}
	try {
		const current = await snapshotPreparedDirectory(path);
		if (current.device === ownership.root.device && current.inode === ownership.root.inode) {
			await rmdir(path);
		}
	} catch {
		// Non-empty or replaced roots are preserved.
	}
}

async function snapshotPreparedTree(path: string, expected: PreparedDirectoryIdentity): Promise<PreparedTreeOwnership> {
	const root = await snapshotPreparedDirectory(path);
	if (root.device !== expected.device || root.inode !== expected.inode) {
		throw new Error("prepared output directory changed");
	}
	const files = new Map<string, PreparedDirectoryIdentity>();
	const directories = new Map<string, PreparedDirectoryIdentity>();
	const visit = async (directory: string, relativeDirectory: string): Promise<void> => {
		for (const name of await readdir(directory)) {
			const relativePath = relativeDirectory.length === 0 ? name : join(relativeDirectory, name);
			const childPath = join(path, relativePath);
			const stats = await lstat(childPath);
			if (stats.isSymbolicLink()) continue;
			const childIdentity = { device: stats.dev, inode: stats.ino };
			if (stats.isFile()) {
				files.set(relativePath, childIdentity);
			} else if (stats.isDirectory()) {
				directories.set(relativePath, childIdentity);
				await visit(childPath, relativePath);
			}
		}
	};
	await visit(path, "");
	return { root, files, directories };
}
