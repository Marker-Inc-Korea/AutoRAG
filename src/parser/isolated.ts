import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";
import { abortReason } from "../process/abort-reason.ts";
import type { DefaultParserRegistryOptions } from "./defaults.ts";
import { ParseError, ParseIsolationError } from "./errors.ts";
import type { ParseInput, ParseOutput } from "./types.ts";

/** Per-file parse deadline when `parserOptions.fileTimeoutMs` is not set. */
export const DEFAULT_PARSE_FILE_TIMEOUT_MS = 120_000;

/**
 * Parsers that run third-party or format-parsing code which can spin, exhaust
 * the heap, or crash on a hostile file. Plain text decoding stays in-process.
 */
export const ISOLATED_PARSER_NAMES: ReadonlySet<string> = new Set(["kordoc", "spreadsheet", "csv", "pptx", "eml"]);

/** The structured-cloneable subset of {@link DefaultParserRegistryOptions} a worker can rebuild its registry from. */
export interface WorkerParserOptions {
	readonly languages?: DefaultParserRegistryOptions["languages"];
	readonly ocr?: DefaultParserRegistryOptions["ocr"];
	readonly kordoc?: { readonly ocr?: boolean | "force" };
}

export interface ParseWorkerRequest {
	readonly id: number;
	readonly parserName: string;
	readonly virtualPath: string;
	readonly sourcePath?: string;
	readonly bytes: Uint8Array;
}

export type ParseWorkerResponse =
	| { readonly id: number; readonly ok: true; readonly output: ParseOutput }
	| {
			readonly id: number;
			readonly ok: false;
			readonly kind: "parse-error";
			readonly parserName: string;
			readonly message: string;
			readonly stack?: string;
	  }
	| {
			readonly id: number;
			readonly ok: false;
			readonly kind: "error";
			readonly message: string;
			readonly stack?: string;
	  };

/**
 * Parser options a worker thread can receive, or `undefined` when they hold a
 * function (an injected `kordoc.parse` or OCR provider) that cannot cross the
 * thread boundary. Such options stay in-process.
 */
export function workerParserOptions(options: DefaultParserRegistryOptions): WorkerParserOptions | undefined {
	const kordoc = options.kordoc;
	if (kordoc?.parse !== undefined || typeof kordoc?.ocr === "function") return undefined;
	return {
		...(options.languages === undefined ? {} : { languages: options.languages }),
		...(options.ocr === undefined ? {} : { ocr: options.ocr }),
		...(kordoc?.ocr === undefined ? {} : { kordoc: { ocr: kordoc.ocr } }),
	};
}

/** Rejects a non-positive or non-finite `fileTimeoutMs` / `maxHeapMb` before any file is parsed. */
export function validateIsolationOptions(
	options: Pick<DefaultParserRegistryOptions, "fileTimeoutMs" | "maxHeapMb">,
): void {
	for (const key of ["fileTimeoutMs", "maxHeapMb"] as const) {
		const value = options[key];
		if (value !== undefined && !(Number.isFinite(value) && value > 0)) {
			throw new RangeError(`parserOptions.${key} must be a positive number, received ${String(value)}`);
		}
	}
}

export interface IsolatedParseRunnerOptions {
	readonly parserOptions: WorkerParserOptions;
	readonly fileTimeoutMs?: number;
	/** V8 `resourceLimits.maxOldGenerationSizeMb` for the worker. Unset keeps Node's default. */
	readonly maxHeapMb?: number;
	/** Override the worker entry (tests). */
	readonly workerUrl?: URL;
}

/**
 * Runs parsers inside a worker thread so one file can be hard-killed. A
 * `Promise.race` deadline cannot interrupt a synchronous spin, but
 * `Worker.terminate()` can; a heap limit turns an out-of-memory parse into a
 * catchable worker error instead of a host crash.
 *
 * One worker serves files in sequence and is reused until it is lost (deadline,
 * out of memory, crash, abort); the next file then gets a fresh worker. Every
 * worker-side loss rejects with a {@link ParseError} whose cause is a
 * {@link ParseIsolationError}, so the caller's existing per-file skip applies.
 */
export class IsolatedParseRunner {
	readonly #options: IsolatedParseRunnerOptions;
	readonly #fileTimeoutMs: number;
	#worker: Worker | undefined;
	#nextId = 0;
	#tail: Promise<unknown> = Promise.resolve();

	constructor(options: IsolatedParseRunnerOptions) {
		this.#options = options;
		this.#fileTimeoutMs = options.fileTimeoutMs ?? DEFAULT_PARSE_FILE_TIMEOUT_MS;
	}

	/** Parse one file. Rejects with the signal's reason when `signal` aborts, after killing the worker. */
	parse(parserName: string, input: ParseInput, signal?: AbortSignal): Promise<ParseOutput> {
		const run = this.#tail.then(() => this.#run(parserName, input, signal));
		this.#tail = run.then(
			() => undefined,
			() => undefined,
		);
		return run;
	}

	/** Terminate the worker, if one is alive. The runner stays usable and spawns a fresh worker on demand. */
	async close(): Promise<void> {
		const worker = this.#worker;
		this.#worker = undefined;
		await worker?.terminate();
	}

	#run(parserName: string, input: ParseInput, signal: AbortSignal | undefined): Promise<ParseOutput> {
		// Executor form: the ES2022 lib this package targets has no Promise.withResolvers.
		return new Promise((resolve, reject) => {
			if (signal?.aborted) {
				reject(abortReason(signal));
				return;
			}
			const worker = this.#worker ?? this.#spawn();
			const id = ++this.#nextId;
			const lost = (code: ParseIsolationError["code"], message: string): void => {
				settle();
				this.#discard(worker);
				reject(new ParseError(parserName, input.virtualPath, new ParseIsolationError(code, message)));
			};
			const onMessage = (response: ParseWorkerResponse): void => {
				if (response.id !== id) return;
				settle();
				if (response.ok) {
					resolve(response.output);
				} else if (response.kind === "parse-error") {
					reject(
						new ParseError(
							response.parserName,
							input.virtualPath,
							Object.assign(
								new Error(response.message),
								response.stack === undefined ? {} : { stack: response.stack },
							),
						),
					);
				} else {
					reject(
						Object.assign(
							new Error(response.message),
							response.stack === undefined ? {} : { stack: response.stack },
						),
					);
				}
			};
			const onError = (error: Error & { code?: string }): void => {
				if (error.code === "ERR_WORKER_OUT_OF_MEMORY") {
					lost("WORKER_OOM", `parse worker ran out of memory: ${error.message}`);
				} else {
					lost("WORKER_CRASHED", `parse worker crashed: ${error.message}`);
				}
			};
			const onExit = (code: number): void => {
				lost("WORKER_CRASHED", `parse worker exited with code ${code} before returning a result`);
			};
			const onAbort = (): void => {
				settle();
				this.#discard(worker);
				reject(abortReason(signal as AbortSignal));
			};
			const timer = setTimeout(() => {
				lost(
					"PARSE_TIMEOUT",
					`parse exceeded the ${this.#fileTimeoutMs}ms per-file deadline and its worker was terminated`,
				);
			}, this.#fileTimeoutMs);
			const settle = (): void => {
				clearTimeout(timer);
				signal?.removeEventListener("abort", onAbort);
				worker.off("message", onMessage);
				worker.off("error", onError);
				worker.off("exit", onExit);
			};
			worker.on("message", onMessage);
			worker.on("error", onError);
			worker.on("exit", onExit);
			signal?.addEventListener("abort", onAbort, { once: true });
			const request: ParseWorkerRequest = {
				id,
				parserName,
				virtualPath: input.virtualPath,
				...(input.sourcePath === undefined ? {} : { sourcePath: input.sourcePath }),
				bytes: input.bytes,
			};
			worker.postMessage(request);
		});
	}

	#spawn(): Worker {
		const worker = new Worker(this.#options.workerUrl ?? defaultWorkerUrl(), {
			workerData: this.#options.parserOptions,
			...(this.#options.maxHeapMb === undefined
				? {}
				: { resourceLimits: { maxOldGenerationSizeMb: this.#options.maxHeapMb } }),
		});
		// An idle worker must not keep the host alive, and an error with no parse in
		// flight must not surface as an unhandled 'error' event.
		worker.unref();
		const forget = (): void => {
			if (this.#worker === worker) this.#worker = undefined;
		};
		worker.on("error", forget);
		worker.on("exit", forget);
		this.#worker = worker;
		return worker;
	}

	#discard(worker: Worker): void {
		if (this.#worker === worker) this.#worker = undefined;
		void worker.terminate();
	}
}

/** The worker entry: `parse-worker.ts` beside this module in source, `parser/parse-worker.js` in a built bundle. */
function defaultWorkerUrl(): URL {
	const here = dirname(fileURLToPath(import.meta.url));
	for (const candidate of [
		join(here, "parse-worker.ts"),
		join(here, "parser", "parse-worker.js"),
		join(here, "..", "parser", "parse-worker.js"),
	]) {
		if (existsSync(candidate)) return pathToFileURL(candidate);
	}
	throw new Error(`The AutoRAG parse worker entry was not found near ${here}.`);
}
