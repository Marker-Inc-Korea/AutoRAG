/**
 * Live E2E: `limits.recordCut` through the real CLI, the real MinSync index and its sparse retrieval.
 *
 * Gated on AUTORAG_RECORD_CUT_LIVE=1 so the offline suite never needs MinSync. Run explicitly with `bun`
 * and a real `minsync` on PATH:
 *
 *   AUTORAG_RECORD_CUT_LIVE=1 bun run vitest run test/live-e2e/record-cut.test.ts
 *
 * The index is sparse-only (`--embedder-id none`), so no embedding model is downloaded or called and no corpus
 * text leaves the machine. Every command runs with AUTORAG_HOME and AUTORAG_CONFIG inside a temp root, so the
 * real ~/.autorag is never read or written.
 *
 * The corpus is one long handbook whose forty sections all match the query, plus three short documents that
 * match it too. Without the cut a top-8 is eight passages of the handbook; with it the short documents
 * come back inside the same top-8.
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const LIVE = process.env.AUTORAG_RECORD_CUT_LIVE === "1";
const REFRESH_TIMEOUT_MS = 600_000;
const CLI = fileURLToPath(new URL("../../src/cli/index.ts", import.meta.url));
const SHORT_DOCUMENTS = ["sku-4471-exception.md", "vendor-memo.md", "audit-note.md"];

interface RetrieveEnvelope {
	readonly results: readonly { readonly source: string; readonly content: string }[];
}

describe.skipIf(!LIVE)("live: record cut keeps a long document from taking every slot", () => {
	let root: string;
	let env: NodeJS.ProcessEnv;
	let configPath: string;

	function cli(args: readonly string[], timeoutMs = 120_000): string {
		const result = spawnSync("bun", ["run", CLI, ...args], { cwd: root, env, encoding: "utf8", timeout: timeoutMs });
		if (result.status !== 0) throw new Error(`autorag ${args.join(" ")} exited ${result.status}: ${result.stderr}`);
		return result.stdout;
	}

	/** Documents hit by a top-k retrieve, counted per file name. */
	function perDocument(topK: number): Record<string, number> {
		const envelope = JSON.parse(
			cli(["lite", "retrieve", "refund approval", "--json", "--top-k", String(topK)]),
		) as RetrieveEnvelope;
		const counts: Record<string, number> = {};
		for (const result of envelope.results)
			counts[basename(result.source)] = (counts[basename(result.source)] ?? 0) + 1;
		return counts;
	}

	function setRecordCut(recordCut: Record<string, unknown> | undefined): void {
		const config = JSON.parse(readFileSync(configPath, "utf8")) as Record<string, unknown>;
		const { limits: _limits, ...rest } = config;
		writeFileSync(
			configPath,
			JSON.stringify(recordCut === undefined ? rest : { ...rest, limits: { recordCut } }, null, 2),
		);
	}

	beforeAll(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-record-cut-live-"));
		const home = join(root, ".autorag-home");
		configPath = join(home, "config.json");
		// The suite setup puts an inert `minsync` first on PATH; this live run needs the real MinSync.
		const path = (process.env.PATH ?? "")
			.split(delimiter)
			.filter((entry) => !entry.includes("autorag-fake-minsync-path-"))
			.join(delimiter);
		// vitest sets NODE_ENV=test, which makes the child CLI refuse to install or resolve a real MinSync.
		env = { ...process.env, NODE_ENV: "production", PATH: path, AUTORAG_HOME: home, AUTORAG_CONFIG: configPath };
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		mkdirSync(join(root, "ws"), { recursive: true });
		mkdirSync(home, { recursive: true });

		const sections = Array.from(
			{ length: 40 },
			(_, index) =>
				`## Section ${index}\n\nRefund approval procedure step ${index}: the refund approval must be recorded and the approval reviewed by finance. Refund approval ledger entry ${index} confirms the refund approval workflow.\n`,
		);
		writeFileSync(join(docs, "handbook-long.md"), `# Refund approval handbook\n\n${sections.join("\n")}`);
		writeFileSync(
			join(docs, "sku-4471-exception.md"),
			"# SKU-4471 exception\n\nRefund approval for SKU-4471 is waived by the regional director, code ZX-9913.\n",
		);
		writeFileSync(
			join(docs, "vendor-memo.md"),
			"# Vendor memo\n\nRefund approval for vendor Kappa uses the alternate form F-212.\n",
		);
		writeFileSync(
			join(docs, "audit-note.md"),
			"# Audit note\n\nThe refund approval audit of Q3 flagged ticket T-8801.\n",
		);

		cli(["init", "--search-paths", docs, "--workspace", join(root, "ws"), "--embedder-id", "none", "--force"]);
		const config = JSON.parse(readFileSync(configPath, "utf8")) as object;
		// The bundled file-name indexers are not what this test covers, and on Windows they outlive the CLI.
		writeFileSync(configPath, JSON.stringify({ ...config, everything: false }, null, 2));
		const refresh = JSON.parse(
			cli(["lite", "refresh", "--method", "parsed,minsync", "--json"], REFRESH_TIMEOUT_MS),
		) as {
			readonly ok: boolean;
			readonly counts?: { readonly written: number };
		};
		expect(refresh.ok, JSON.stringify(refresh)).toBe(true);
		expect(refresh.counts?.written).toBe(4);
	}, REFRESH_TIMEOUT_MS + 120_000);

	afterAll(() => rmSync(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 500 }));

	it("fills a plain top-8 with one document's passages when the cut is off", () => {
		setRecordCut(undefined);
		expect(perDocument(8)).toEqual({ "handbook-long.md": 8 });
	});

	it("returns every short document inside the same top-8 when the cut is on", () => {
		setRecordCut({ enabled: true });
		const counts = perDocument(8);

		for (const name of SHORT_DOCUMENTS)
			expect(counts[name], `${name} missing from ${JSON.stringify(counts)}`).toBe(1);
		expect(counts["handbook-long.md"]).toBe(5);
		expect(Object.values(counts).reduce((sum, count) => sum + count, 0)).toBe(8);
	});

	it("honours a tuned newRecordShare of 0: the best passages win, so the long document is capped but not displaced", () => {
		setRecordCut({ enabled: true, newRecordShare: 0, perRecordFloor: 6, perRecordDivisor: 1 });
		expect(perDocument(8)["handbook-long.md"]).toBeGreaterThanOrEqual(6);
	});

	it("rejects a malformed recordCut section with the field named", () => {
		setRecordCut({ enabled: true, depth: 99 });
		expect(() => cli(["lite", "retrieve", "refund approval", "--json"])).toThrow(
			/limits\.recordCut\.depth must be an integer from 1 to 10/u,
		);
	});
});
