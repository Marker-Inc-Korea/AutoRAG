import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createDefaultParserRegistry, ImageOcrParser, ParseError } from "../../src/parser/index.ts";

const tesseractMock = vi.hoisted(() => ({
	createWorker: vi.fn(),
}));

vi.mock("tesseract.js", () => tesseractMock);

describe("ImageOcrParser", () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
	});

	it("is opt-in and enforces timeout and maxBytes budgets", async () => {
		// Given: the default registry and explicitly enabled OCR registries with budget controls.
		let abortObserved = false;
		const disabled = createDefaultParserRegistry();
		const timed = createDefaultParserRegistry({
			ocr: {
				enabled: true,
				timeoutMs: 1,
				engine: (input) =>
					new Promise<string>(() => {
						input.signal.addEventListener("abort", () => {
							abortObserved = true;
						});
					}),
			},
		});
		const budgeted = createDefaultParserRegistry({
			ocr: {
				enabled: true,
				maxBytes: 3,
				engine: async () => "OCR marker",
			},
		});

		// When/Then: images are invisible by default but become routed when OCR is explicitly enabled.
		expect(disabled.getForVirtualPath("/docs/scan.png")).toBeUndefined();
		const imageParser = timed.getForVirtualPath("/docs/scan.png");
		expect(imageParser).toBeDefined();
		const timeoutResult = imageParser?.parse({
			virtualPath: "/docs/scan.png",
			bytes: Buffer.from([0x89, 0x50, 0x4e, 0x47]),
		});
		const timeoutAssertion = expect(timeoutResult).rejects.toThrow(/timed out/i);
		await vi.advanceTimersByTimeAsync(1);
		await timeoutAssertion;
		expect(abortObserved).toBe(true);
		await expect(
			budgeted.getForVirtualPath("/docs/large.png")?.parse({
				virtualPath: "/docs/large.png",
				bytes: Buffer.from([0x89, 0x50, 0x4e, 0x47]),
			}),
		).rejects.toBeInstanceOf(ParseError);
	});

	it("waits for injected engine cleanup before returning on timeout", async () => {
		// Given: an OCR engine that starts cleanup only after the abort signal.
		let cleanupCompleted = false;
		const parser = new ImageOcrParser({
			enabled: true,
			timeoutMs: 1,
			engine: (input) =>
				new Promise<string>(() => {
					input.signal.addEventListener("abort", () => {
						cleanupCompleted = true;
					});
				}),
		});

		// When: parsing times out.
		const result = parser.parse({ virtualPath: "/docs/scan.png", bytes: Buffer.from([0x89, 0x50]) });
		const assertion = expect(result).rejects.toBeInstanceOf(ParseError);
		await vi.advanceTimersByTimeAsync(1);
		await assertion;

		// Then: cleanup has completed before parse() resolves/rejects.
		expect(cleanupCompleted).toBe(true);
	});

	it("waits for Tesseract worker termination before returning on timeout", async () => {
		// Given: the real OCR adapter observes a timeout while Tesseract termination is still pending.
		let finishTermination: () => void = () => undefined;
		tesseractMock.createWorker.mockResolvedValueOnce({
			recognize: () => new Promise<never>(() => undefined),
			terminate: () =>
				new Promise<void>((resolve) => {
					finishTermination = resolve;
				}),
		});
		const parser = new ImageOcrParser({ enabled: true, timeoutMs: 1 });
		const result = observeSettlement(
			parser.parse({ virtualPath: "/docs/scan.png", bytes: Buffer.from([0x89, 0x50]) }),
		);
		await Promise.resolve();

		// When: timeout fires but worker termination has not completed.
		await vi.advanceTimersByTimeAsync(1);
		await Promise.resolve();
		expect(result.settled()).toBe(false);

		// Then: parse() rejects only after terminate() completes.
		finishTermination();
		await expect(result.promise).rejects.toBeInstanceOf(ParseError);
		expect(result.settled()).toBe(true);
	});

	it("waits for pending Tesseract worker creation cleanup", async () => {
		// Given: Tesseract worker creation is still pending when the OCR timeout fires.
		let resolveWorker: (worker: { recognize: () => Promise<string>; terminate: () => Promise<void> }) => void = () =>
			undefined;
		let finishTermination: () => void = () => undefined;
		tesseractMock.createWorker.mockReturnValueOnce(
			new Promise((resolve) => {
				resolveWorker = resolve;
			}),
		);
		const parser = new ImageOcrParser({ enabled: true, timeoutMs: 1 });
		const result = observeSettlement(
			parser.parse({ virtualPath: "/docs/scan.png", bytes: Buffer.from([0x89, 0x50]) }),
		);

		// When: timeout fires before createWorker resolves.
		await vi.advanceTimersByTimeAsync(1);
		await Promise.resolve();
		expect(result.settled()).toBe(false);

		// Then: parse() rejects only after the late-created worker is terminated.
		resolveWorker({
			recognize: async () => "late text",
			terminate: () =>
				new Promise<void>((resolve) => {
					finishTermination = resolve;
				}),
		});
		await Promise.resolve();
		expect(result.settled()).toBe(false);
		finishTermination();
		await expect(result.promise).rejects.toBeInstanceOf(ParseError);
		expect(result.settled()).toBe(true);
	});
});

function observeSettlement<T>(promise: Promise<T>): { readonly promise: Promise<T>; readonly settled: () => boolean } {
	let settled = false;
	promise.then(
		() => {
			settled = true;
		},
		() => {
			settled = true;
		},
	);
	return { promise, settled: () => settled };
}
