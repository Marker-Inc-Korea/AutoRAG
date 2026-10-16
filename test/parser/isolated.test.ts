import { afterEach, describe, expect, it } from "vitest";
import { ParseError } from "../../src/parser/index.ts";
import { IsolatedParseRunner } from "../../src/parser/isolated.ts";
import { createDocxFixture } from "../fixtures/document-formats.ts";

const hostileWorker = new URL("../fixtures/parse-workers/hostile-worker.ts", import.meta.url);
const bytes = new Uint8Array([1, 2, 3]);

const runners: IsolatedParseRunner[] = [];
function makeRunner(options: ConstructorParameters<typeof IsolatedParseRunner>[0]): IsolatedParseRunner {
	const runner = new IsolatedParseRunner(options);
	runners.push(runner);
	return runner;
}

afterEach(async () => {
	await Promise.all(runners.splice(0).map((runner) => runner.close()));
});

function causeCode(error: unknown): unknown {
	return error instanceof ParseError && error.cause instanceof Error
		? (error.cause as Error & { code?: string }).code
		: undefined;
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
	try {
		await promise;
	} catch (error) {
		return error;
	}
	throw new Error("expected the promise to reject");
}

describe("IsolatedParseRunner", () => {
	it("parses a real document in the worker and reuses it for the next file", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 30_000 });
		const docx = await createDocxFixture("Isolated worker marker alpha");
		const first = await runner.parse("kordoc", { virtualPath: "/docs/a.docx", bytes: docx });
		const second = await runner.parse("kordoc", { virtualPath: "/docs/b.docx", bytes: docx });
		expect(first.markdown).toContain("Isolated worker marker alpha");
		expect(second.markdown).toContain("Isolated worker marker alpha");
	});

	it("hard-kills a CPU-bound parse at the deadline and recovers for the next file", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 200, workerUrl: hostileWorker });
		const started = Date.now();
		const error = await rejection(runner.parse("kordoc", { virtualPath: "/docs/spin.bin", bytes }));
		const elapsed = Date.now() - started;

		expect(error).toBeInstanceOf(ParseError);
		expect(causeCode(error)).toBe("PARSE_TIMEOUT");
		expect((error as ParseError).virtualPath).toBe("/docs/spin.bin");
		expect((error as ParseError).message).toContain("200ms");
		// A Promise.race timeout cannot interrupt a synchronous spin; terminate() can.
		expect(elapsed).toBeLessThan(3_000);

		const next = await runner.parse("kordoc", { virtualPath: "/docs/ok.bin", bytes });
		expect(next.markdown).toBe("fine /docs/ok.bin");
	});

	it("turns a worker heap OOM into a ParseError while the host survives", async () => {
		const runner = makeRunner({
			parserOptions: {},
			fileTimeoutMs: 30_000,
			maxHeapMb: 24,
			workerUrl: hostileWorker,
		});
		const error = await rejection(runner.parse("kordoc", { virtualPath: "/docs/oom.bin", bytes }));
		expect(error).toBeInstanceOf(ParseError);
		expect(causeCode(error)).toBe("WORKER_OOM");

		const next = await runner.parse("kordoc", { virtualPath: "/docs/ok.bin", bytes });
		expect(next.markdown).toBe("fine /docs/ok.bin");
	});

	it("turns an uncaught worker exception and an unexpected exit into ParseErrors", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 30_000, workerUrl: hostileWorker });
		const crashed = await rejection(runner.parse("eml", { virtualPath: "/docs/crash.bin", bytes }));
		expect(crashed).toBeInstanceOf(ParseError);
		expect((crashed as ParseError).parserName).toBe("eml");
		expect(causeCode(crashed)).toBe("WORKER_CRASHED");
		expect(((crashed as ParseError).cause as Error).message).toContain("crash sentinel");

		const exited = await rejection(runner.parse("pptx", { virtualPath: "/docs/exit.bin", bytes }));
		expect(exited).toBeInstanceOf(ParseError);
		expect(causeCode(exited)).toBe("WORKER_CRASHED");
		expect(((exited as ParseError).cause as Error).message).toContain("code 3");
	});

	it("preserves the parser's own root cause verbatim", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 30_000, workerUrl: hostileWorker });
		const error = await rejection(runner.parse("kordoc", { virtualPath: "/docs/rooted.bin", bytes }));
		expect(error).toBeInstanceOf(ParseError);
		expect((error as ParseError).message).toContain("root cause sentinel");
		expect(((error as ParseError).cause as Error).stack).toContain("at fixture");
	});

	it("reports a real corrupt document as a ParseError carrying kordoc's message", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 30_000 });
		const error = await rejection(
			runner.parse("kordoc", { virtualPath: "/docs/bad.docx", bytes: new Uint8Array([9, 9, 9, 9]) }),
		);
		expect(error).toBeInstanceOf(ParseError);
		expect((error as ParseError).message.length).toBeGreaterThan(
			'Parser "kordoc" failed for /docs/bad.docx: '.length,
		);
	});

	it("kills an in-flight parse when the signal aborts and rejects with the abort reason", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 30_000, workerUrl: hostileWorker });
		const controller = new AbortController();
		const pending = rejection(runner.parse("kordoc", { virtualPath: "/docs/spin.bin", bytes }, controller.signal));
		setTimeout(() => controller.abort(), 150);
		const started = Date.now();
		const error = await pending;
		expect(error).not.toBeInstanceOf(ParseError);
		expect((error as Error).name).toBe("AbortError");
		expect(Date.now() - started).toBeLessThan(3_000);
	});

	it("rejects immediately when the signal is already aborted", async () => {
		const runner = makeRunner({ parserOptions: {}, fileTimeoutMs: 30_000, workerUrl: hostileWorker });
		const error = await rejection(
			runner.parse("kordoc", { virtualPath: "/docs/ok.bin", bytes }, AbortSignal.abort()),
		);
		expect((error as Error).name).toBe("AbortError");
	});
});
