import { describe, expect, it, vi } from "vitest";
import {
	type FetchProvider,
	isLowQualityOutput,
	parseJinaReaderContent,
	renderHtmlToText,
} from "../../src/web/fetch/html-renderer.ts";

const substantial = (label: string) => `${label} ${"useful article content ".repeat(8)}`;

describe("renderHtmlToText", () => {
	it("tries injected backends in native, lynx, firecrawl, jina order", async () => {
		const calls: FetchProvider[] = [];
		const runners = Object.fromEntries(
			(["native", "lynx", "firecrawl", "jina"] as const).map((provider) => [
				provider,
				vi.fn(async () => {
					calls.push(provider);
					return provider === "jina" ? substantial("jina") : null;
				}),
			]),
		);

		const result = await renderHtmlToText("https://example.test", "<html></html>", {
			timeoutSeconds: 1,
			runners,
		});
		expect(calls).toEqual(["native", "lynx", "firecrawl", "jina"]);
		expect(result).toMatchObject({ ok: true, method: "jina" });
	});

	it("rejects output with 100 or fewer non-whitespace characters", async () => {
		const result = await renderHtmlToText("https://example.test", "<html></html>", {
			runners: {
				native: async () => `${"x ".repeat(100)}`,
				lynx: async () => substantial("lynx"),
				firecrawl: async () => null,
				jina: async () => null,
			},
		});
		expect(result.method).toBe("lynx");
	});

	it("demotes low-quality output and returns the first good fallback", async () => {
		const navigation = `${Array.from({ length: 12 }, (_, index) => `Menu ${index}`).join("\n")}\n${"x".repeat(120)}`;
		const result = await renderHtmlToText("https://example.test", "<html></html>", {
			runners: {
				native: async () => navigation,
				lynx: async () => substantial("article"),
				firecrawl: async () => null,
				jina: async () => null,
			},
		});
		expect(isLowQualityOutput(navigation)).toBe(true);
		expect(result.method).toBe("lynx");
	});

	it("returns the best demoted output when every substantial backend is low quality", async () => {
		const gated = `Please enable JavaScript ${"x".repeat(120)}`;
		const result = await renderHtmlToText("https://example.test", "<html></html>", {
			runners: {
				native: async () => gated,
				lynx: async () => null,
				firecrawl: async () => null,
				jina: async () => null,
			},
		});
		expect(result).toMatchObject({ ok: true, method: "native", content: gated });
	});
});

describe("parseJinaReaderContent", () => {
	it("extracts substantial content after the marker", () => {
		const markdown = substantial("# Heading").trimEnd();
		expect(parseJinaReaderContent(`Title: Example\nMarkdown Content:\n${markdown}`)).toBe(markdown);
	});

	it("rejects missing, loading, and short marker content", () => {
		expect(parseJinaReaderContent(substantial("unmarked"))).toBeNull();
		expect(parseJinaReaderContent(`Markdown Content:\nLoading... ${"x".repeat(120)}`)).toBeNull();
		expect(parseJinaReaderContent("Markdown Content:\nshort")).toBeNull();
	});
});
