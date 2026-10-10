import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createTesseractOcrProvider } from "../../src/parser/ocr-engines.ts";

const tesseractMock = vi.hoisted(() => ({
	createWorker: vi.fn(),
}));

vi.mock("tesseract.js", () => tesseractMock);

const PNG_HEADER = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);

describe("Tesseract OCR provider lifecycle", () => {
	beforeEach(() => {
		vi.useFakeTimers();
		tesseractMock.createWorker.mockReset();
	});

	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
	});

	it("returns the recognized text and terminates the worker", async () => {
		const terminate = vi.fn(async () => undefined);
		tesseractMock.createWorker.mockResolvedValueOnce({
			recognize: async () => ({ data: { text: "recognized" } }),
			terminate,
		});
		const provider = createTesseractOcrProvider({ languages: ["ko", "en"] });

		await expect(provider(PNG_HEADER, 1, "image/png")).resolves.toBe("recognized");

		expect(tesseractMock.createWorker).toHaveBeenCalledWith("kor+eng", undefined, {});
		expect(terminate).toHaveBeenCalledOnce();
	});

	it("waits for Tesseract worker termination before rejecting on timeout", async () => {
		// Given: recognition never finishes and worker termination is still pending when the timeout fires.
		let finishTermination: () => void = () => undefined;
		tesseractMock.createWorker.mockResolvedValueOnce({
			recognize: () => new Promise<never>(() => undefined),
			terminate: () =>
				new Promise<void>((resolve) => {
					finishTermination = resolve;
				}),
		});
		const provider = createTesseractOcrProvider({ languages: ["en"], timeoutMs: 1 });
		const result = observeSettlement(provider(PNG_HEADER, 1, "image/png"));
		await Promise.resolve();

		// When: the timeout fires but termination has not completed.
		await vi.advanceTimersByTimeAsync(1);
		await Promise.resolve();
		expect(result.settled()).toBe(false);

		// Then: the call rejects only after terminate() completes.
		finishTermination();
		await expect(result.promise).rejects.toThrow(/timed out/i);
		expect(result.settled()).toBe(true);
	});

	it("waits for pending Tesseract worker creation cleanup", async () => {
		// Given: worker creation is still pending when the timeout fires.
		let resolveWorker: (worker: { recognize: () => Promise<never>; terminate: () => Promise<void> }) => void = () =>
			undefined;
		let finishTermination: () => void = () => undefined;
		tesseractMock.createWorker.mockReturnValueOnce(
			new Promise((resolve) => {
				resolveWorker = resolve;
			}),
		);
		const provider = createTesseractOcrProvider({ languages: ["en"], timeoutMs: 1 });
		const result = observeSettlement(provider(PNG_HEADER, 1, "image/png"));

		// When: the timeout fires before createWorker resolves.
		await vi.advanceTimersByTimeAsync(1);
		await Promise.resolve();
		expect(result.settled()).toBe(false);

		// Then: the call rejects only after the late-created worker is terminated.
		resolveWorker({
			recognize: () => new Promise<never>(() => undefined),
			terminate: () =>
				new Promise<void>((resolve) => {
					finishTermination = resolve;
				}),
		});
		await Promise.resolve();
		expect(result.settled()).toBe(false);
		finishTermination();
		await expect(result.promise).rejects.toThrow(/timed out/i);
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
