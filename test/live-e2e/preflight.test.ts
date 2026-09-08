import { describe, expect, it } from "vitest";
import { runPreflight } from "../../scripts/live-e2e/preflight.mjs";

type FetchResponse = { readonly ok: boolean; readonly status: number; json: () => Promise<unknown> };
const response = (body: unknown, status = 200): FetchResponse => ({
	ok: status >= 200 && status < 300,
	status,
	json: async () => body,
});
const readyFetch = async (): Promise<FetchResponse> =>
	response([[0.1, 0.2, 0.3, ...Array.from({ length: 765 }, () => 0)]]);

const optionalBinaries = Object.freeze({
	katok: false,
	discrawl: false,
	qmd: false,
	rclone: false,
	mailcrawl: false,
	jikji: false,
	minsync: false,
});

describe("live-e2e preflight", () => {
	it("refuses a non-loopback embedding endpoint before probing it", async () => {
		const result = await runPreflight({
			endpoint: "https://example.test/embed",
			fetchImpl: readyFetch,
			binaries: optionalBinaries,
		});
		expect(result.verdict).toBe("refused");
		expect(result.code).toBe("live-e2e-non-loopback-embedding");
	});

	it("refuses OpenAI egress without echoing the credential", async () => {
		const result = await runPreflight({
			endpoint: "http://127.0.0.1:18080",
			fetchImpl: readyFetch,
			binaries: optionalBinaries,
			openaiKey: "fake-secret-value",
		});
		const serialized = JSON.stringify(result);
		expect(result.verdict).toBe("refused");
		expect(result.code).toBe("live-e2e-openai-egress");
		expect(result.openaiKeyPresent).toBe(false);
		expect(serialized).not.toContain("fake-secret-value");
	});

	it("skips missing optional binaries but fails an explicitly configured lane", async () => {
		const skipped = await runPreflight({
			endpoint: "http://127.0.0.1:18080",
			fetchImpl: readyFetch,
			binaries: optionalBinaries,
		});
		expect(skipped.verdict).toBe("degraded");
		expect(skipped.lanes.katok.status).toBe("SKIP");
		const failed = await runPreflight({
			endpoint: "http://127.0.0.1:18080",
			fetchImpl: readyFetch,
			binaries: optionalBinaries,
			configuredLanes: ["katok"],
		});
		expect(failed.verdict).toBe("refused");
		expect(failed.lanes.katok.status).toBe("FAIL");
	});

	it("returns ready with zero exit code when the local service and all lanes are ready", async () => {
		const binaries = Object.freeze(Object.fromEntries(Object.keys(optionalBinaries).map((binary) => [binary, true])));
		const result = await runPreflight({
			endpoint: "http://localhost:18080",
			fetchImpl: readyFetch,
			binaries,
			configuredLanes: ["minsync"],
		});
		expect(result.verdict).toBe("ready");
		expect(result.exitCode).toBe(0);
		expect(result.embedding.dimension).toBe(768);
	});
});
