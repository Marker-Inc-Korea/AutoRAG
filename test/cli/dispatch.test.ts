import { afterEach, describe, expect, it, vi } from "vitest";
import { main, parseArgs } from "../../src/cli/index.ts";

describe("parseArgs", () => {
	it("collects positionals in order", () => {
		const parsed = parseArgs(["search", "hello", "world"]);
		expect("error" in parsed).toBe(false);
		if ("error" in parsed) return;
		expect(parsed.positionals).toEqual(["search", "hello", "world"]);
	});

	it("parses boolean and value flags", () => {
		const parsed = parseArgs(["refresh", "--force", "--config", "a.json", "--json"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags.force).toBe(true);
		expect(parsed.flags.json).toBe(true);
		expect(parsed.flags.config).toBe("a.json");
	});

	it("parses --key=value form", () => {
		const parsed = parseArgs(["search", "--top-k=5"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["top-k"]).toBe("5");
	});

	it("keeps two-word command sub-words as positionals", () => {
		const parsed = parseArgs(["memory", "inspect"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.positionals).toEqual(["memory", "inspect"]);
	});

	it("rejects an unknown flag", () => {
		const parsed = parseArgs(["refresh", "--nope"]);
		expect(parsed).toEqual({ error: "Unknown flag: --nope" });
	});

	it("rejects a value flag without a value", () => {
		const parsed = parseArgs(["search", "--scope"]);
		expect("error" in parsed).toBe(true);
	});

	it("accepts --method as a value flag", () => {
		const parsed = parseArgs(["refresh", "--method", "bm25,minsync"]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["method"]).toBe("bm25,minsync");
	});

	it("accepts embedder-* value flags", () => {
		const parsed = parseArgs([
			"init",
			"--embedder-id",
			"text-embedding-3-small",
			"--embedder-dimension",
			"1536",
			"--embedder-api-key-env",
			"OPENAI_API_KEY",
		]);
		if ("error" in parsed) throw new Error(parsed.error);
		expect(parsed.flags["embedder-id"]).toBe("text-embedding-3-small");
		expect(parsed.flags["embedder-dimension"]).toBe("1536");
		expect(parsed.flags["embedder-api-key-env"]).toBe("OPENAI_API_KEY");
	});
});

describe("main routing", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("prints usage and exits 0 for --help", async () => {
		const out = vi.spyOn(process.stdout, "write").mockReturnValue(true);
		const code = await main(["--help"]);
		expect(code).toBe(0);
		expect(out).toHaveBeenCalled();
		expect(String(out.mock.calls[0]?.[0])).toContain("Usage: autorag");
	});

	it("prints usage and exits 0 with no command", async () => {
		vi.spyOn(process.stdout, "write").mockReturnValue(true);
		expect(await main([])).toBe(0);
	});

	it("exits 2 for an unknown command", async () => {
		const err = vi.spyOn(process.stderr, "write").mockReturnValue(true);
		const code = await main(["frobnicate"]);
		expect(code).toBe(2);
		expect(String(err.mock.calls[0]?.[0])).toContain("Unknown command");
	});

	it("exits 2 for an unknown flag", async () => {
		vi.spyOn(process.stderr, "write").mockReturnValue(true);
		expect(await main(["refresh", "--bogus"])).toBe(2);
	});
});
