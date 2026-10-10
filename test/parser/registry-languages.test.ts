import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createDefaultParserRegistry, resolveParserOptions } from "../../src/parser/defaults.ts";
import type { KordocParseCallOptions, KordocParseFn } from "../../src/parser/kordoc.ts";

const tesseractMock = vi.hoisted(() => ({ createWorker: vi.fn() }));
vi.mock("tesseract.js", () => tesseractMock);

/** Stub kordoc: records the `ocr` option and, when it is a provider, runs it like kordoc does per page. */
function createKordocStub() {
	const captured: Array<KordocParseCallOptions["ocr"]> = [];
	const parse: KordocParseFn = async (_bytes, options) => {
		captured.push(options.ocr);
		const markdown =
			typeof options.ocr === "function" ? await options.ocr(new Uint8Array([1]), 1, "image/png") : "body";
		return { success: true, fileType: "pdf", markdown };
	};
	return { captured, parse };
}

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"] as const;

describe("createDefaultParserRegistry image ownership", () => {
	it("leaves image files unparsed unless OCR is enabled", () => {
		const registry = createDefaultParserRegistry();

		for (const extension of IMAGE_EXTENSIONS) {
			expect(registry.getForVirtualPath(`/docs/shot${extension}`)).toBeUndefined();
		}
	});

	it("routes png, jpg, jpeg and webp to the single kordoc parser once OCR is enabled", () => {
		const registry = createDefaultParserRegistry({ ocr: { enabled: true } });
		const pdfParser = registry.getForVirtualPath("/docs/a.pdf");

		for (const extension of IMAGE_EXTENSIONS) {
			const parser = registry.getForVirtualPath(`/docs/shot${extension}`);
			expect(parser?.name).toBe("kordoc");
			expect(parser).toBe(pdfParser);
		}
	});

	it("does not claim bmp or tiff, which kordoc rejects as unsupported", () => {
		const registry = createDefaultParserRegistry({ ocr: { enabled: true } });

		expect(registry.getForVirtualPath("/docs/a.bmp")).toBeUndefined();
		expect(registry.getForVirtualPath("/docs/a.tiff")).toBeUndefined();
	});
});

describe("createDefaultParserRegistry OCR engine selection", () => {
	beforeEach(() => {
		tesseractMock.createWorker.mockReset();
		tesseractMock.createWorker.mockResolvedValue({
			recognize: vi.fn(async () => ({ data: { text: "tesseract text" } })),
			terminate: vi.fn(async () => undefined),
		});
	});

	it.each([["ko"], ["en"], ["ko", "en"]] as const)("uses the built-in engine for %s only", async (...languages) => {
		const stub = createKordocStub();
		const registry = createDefaultParserRegistry({
			languages,
			ocr: { enabled: true },
			kordoc: { parse: stub.parse },
		});

		await registry
			.getForVirtualPath("/docs/a.pdf")
			?.parse({ virtualPath: "/docs/a.pdf", bytes: new Uint8Array([1]) });
		await registry
			.getForVirtualPath("/docs/a.png")
			?.parse({ virtualPath: "/docs/a.png", bytes: new Uint8Array([1]) });

		expect(stub.captured).toEqual([true, true]);
		expect(tesseractMock.createWorker).not.toHaveBeenCalled();
	});

	it.each([
		[["ja"], "jpn"],
		[["zh-hans"], "chi_sim"],
		[["ko", "ja"], "kor+jpn"],
		[["ko", "en", "ru"], "kor+eng+rus"],
		// Measured: the built-in model misreads accents (fr 0.64, es 0.45), Tesseract does not.
		[["fr"], "fra"],
		[["ko", "de"], "kor+deu"],
	] as const)("injects Tesseract for %j (%s)", async (languages, traineddata) => {
		const stub = createKordocStub();
		const registry = createDefaultParserRegistry({
			languages,
			ocr: { enabled: true },
			kordoc: { parse: stub.parse },
		});

		const parsed = await registry
			.getForVirtualPath("/docs/a.png")
			?.parse({ virtualPath: "/docs/a.png", bytes: new Uint8Array([1]) });

		expect(typeof stub.captured[0]).toBe("function");
		expect(parsed?.markdown).toContain("tesseract text");
		expect(tesseractMock.createWorker).toHaveBeenCalledWith(traineddata, undefined, expect.any(Object));
	});

	it("never runs OCR for any language while the switch is off", async () => {
		const stub = createKordocStub();
		const registry = createDefaultParserRegistry({ languages: ["ja"], kordoc: { parse: stub.parse } });

		await registry
			.getForVirtualPath("/docs/a.pdf")
			?.parse({ virtualPath: "/docs/a.pdf", bytes: new Uint8Array([1]) });

		expect(stub.captured).toEqual([undefined]);
		expect(tesseractMock.createWorker).not.toHaveBeenCalled();
	});

	it("lets an explicit kordoc.ocr setting win over language-based selection", async () => {
		const stub = createKordocStub();
		const registry = createDefaultParserRegistry({
			languages: ["ja"],
			ocr: { enabled: true },
			kordoc: { parse: stub.parse, ocr: "force" },
		});

		await registry
			.getForVirtualPath("/docs/a.pdf")
			?.parse({ virtualPath: "/docs/a.pdf", bytes: new Uint8Array([1]) });

		expect(stub.captured).toEqual(["force"]);
	});
});

describe("createDefaultParserRegistry Tesseract cache", () => {
	let cacheRoot: string;

	beforeEach(() => {
		cacheRoot = mkdtempSync(join(tmpdir(), "autorag-tessdata-"));
		tesseractMock.createWorker.mockReset();
		tesseractMock.createWorker.mockResolvedValue({
			recognize: vi.fn(async () => ({ data: { text: "text" } })),
			terminate: vi.fn(async () => undefined),
		});
	});

	afterEach(() => {
		rmSync(cacheRoot, { recursive: true, force: true });
	});

	it("hands the configured cache directory to Tesseract instead of the working directory", async () => {
		const cachePath = join(cacheRoot, "nested", "tessdata");
		const registry = createDefaultParserRegistry({
			languages: ["ja"],
			ocr: { enabled: true, cachePath },
			kordoc: { parse: createKordocStub().parse },
		});

		await registry
			.getForVirtualPath("/docs/a.png")
			?.parse({ virtualPath: "/docs/a.png", bytes: new Uint8Array([1]) });

		expect(tesseractMock.createWorker).toHaveBeenCalledWith("jpn", undefined, { cachePath });
	});
});

describe("resolveParserOptions", () => {
	it("fills the registry languages from the configured global languages", () => {
		expect(resolveParserOptions(undefined, ["ja", "en"])).toEqual({ languages: ["ja", "en"] });
	});

	it("keeps caller-provided parser options and adds languages", () => {
		const resolved = resolveParserOptions({ ocr: { enabled: true } }, ["ko"]);
		expect(resolved).toEqual({ ocr: { enabled: true }, languages: ["ko"] });
	});

	it("lets an explicit parser-level language list win", () => {
		const resolved = resolveParserOptions({ languages: ["th"] }, ["ko", "en"]);
		expect(resolved.languages).toEqual(["th"]);
	});

	it("pins the Tesseract cache under the workspace when OCR options are present", () => {
		const resolved = resolveParserOptions({ ocr: { enabled: true } }, ["ko"], "/work/space");
		expect(resolved.ocr?.cachePath).toBe(join("/work/space", ".autorag", "models", "tessdata"));
	});

	it("keeps an explicit cache path and does not invent OCR options", () => {
		const explicit = resolveParserOptions({ ocr: { enabled: true, cachePath: "/custom" } }, ["ko"], "/work/space");
		expect(explicit.ocr?.cachePath).toBe("/custom");
		expect(resolveParserOptions(undefined, ["ko"], "/work/space").ocr).toBeUndefined();
	});
});
