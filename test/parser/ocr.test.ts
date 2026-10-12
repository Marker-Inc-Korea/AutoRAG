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

		expect(tesseractMock.createWorker).toHaveBeenCalledWith("kor+eng", undefined, {
			errorHandler: expect.any(Function),
		});
		expect(terminate).toHaveBeenCalledOnce();
	});

	it("contains a worker image error instead of an uncaught exception", async () => {
		vi.useRealTimers();
		const terminate = vi.fn(async () => undefined);
		const uncaught: unknown[] = [];
		const onUncaught = (error: unknown) => {
			uncaught.push(error);
		};
		tesseractMock.createWorker.mockImplementationOnce(
			(_language: unknown, _oem: unknown, options: { errorHandler?: (data: unknown) => void }) => ({
				recognize: () => {
					const payload = "Error attempting to read image";
					// createWorker.js rejects the job, then throws from the message handler unless errorHandler is set.
					process.nextTick(() => {
						if (typeof options?.errorHandler === "function") options.errorHandler(payload);
						else throw new Error(payload);
					});
					return Promise.reject(new Error(payload));
				},
				terminate,
			}),
		);
		process.on("uncaughtException", onUncaught);
		try {
			const provider = createTesseractOcrProvider({ languages: ["ja"] });
			await expect(provider(PNG_HEADER, 1, "image/png")).rejects.toThrow("Error attempting to read image");
			await new Promise((resolve) => process.nextTick(resolve));
			expect(terminate).toHaveBeenCalledOnce();
			expect(uncaught).toEqual([]);
		} finally {
			process.off("uncaughtException", onUncaught);
		}
	});

	it("rejects a 2x36 image before starting a worker", async () => {
		const provider = createTesseractOcrProvider({ languages: ["ja"] });

		await expect(provider(pngSize(2, 36), 1, "image/png")).rejects.toThrow(/2x36/);
		expect(tesseractMock.createWorker).not.toHaveBeenCalled();
	});

	it("still recognizes images that are not below the 3 pixel minimum", async () => {
		const recognize = vi.fn(async () => ({ data: { text: "ok" } }));
		tesseractMock.createWorker.mockResolvedValue({
			recognize,
			terminate: async () => undefined,
		});
		const provider = createTesseractOcrProvider({ languages: ["ja"] });

		await expect(provider(pngSize(12, 907), 1, "image/png")).resolves.toBe("ok");
		await expect(provider(PNG_HEADER, 1, "image/png")).resolves.toBe("ok");
		await expect(provider(new Uint8Array([1, 2, 3, 4]), 1, "image/png")).resolves.toBe("ok");
		expect(recognize).toHaveBeenCalledTimes(3);
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

/** PNG signature plus an IHDR chunk, which is all the dimension check reads. */
function pngSize(width: number, height: number): Uint8Array {
	const bytes = new Uint8Array(24);
	bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52]);
	const view = new DataView(bytes.buffer);
	view.setUint32(16, width);
	view.setUint32(20, height);
	return bytes;
}

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
