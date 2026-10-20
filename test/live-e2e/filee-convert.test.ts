/**
 * Live E2E: the real `filee` and `dupey` binaries behind `convert_files`.
 *
 * Gated on AUTORAG_FILEE_LIVE=1 so the offline suite never needs either
 * program. Run explicitly, with `filee` (or `AUTORAG_FILEE_BIN`) and `dupey` on PATH:
 *
 *   AUTORAG_FILEE_LIVE=1 bun run vitest run test/live-e2e/filee-convert.test.ts
 *
 * Proves what the unit tests can only fake: the argv Filee accepts, its JSON
 * contract, multibyte (Korean) paths and text, `--overwrite`, and that
 * converted copies land in dupey families exactly as `dupey scan` reports.
 */
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { convertAndRelate, convertWithFilee, type FileeCliOptions, listFileeFormats } from "../../src/convert/index.ts";
import { scanWithDupey } from "../../src/dupey/index.ts";

const LIVE = process.env.AUTORAG_FILEE_LIVE === "1";
const filee: FileeCliOptions = process.env.AUTORAG_FILEE_BIN ? { executable: process.env.AUTORAG_FILEE_BIN } : {};

const REPORT = `# 분기 보고서

이번 분기 매출은 전년 대비 12% 증가했습니다. 주요 요인은 신규 고객 확보와 기존 고객의 재구매율 상승입니다.

## 세부 내용

- 신규 고객 320명
- 재구매율 41%
- 해지율 3.2%

결론적으로 다음 분기에도 성장세를 유지할 것으로 전망합니다. 추가 투자는 마케팅과 고객 지원에 집중합니다.
`;

describe.skipIf(!LIVE)("live: convert_files with real filee + dupey", () => {
	let root: string;
	let dataDir: string;
	beforeAll(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-filee-live-"));
		dataDir = join(root, "filee-data");
		mkdirSync(join(root, "문서 폴더"), { recursive: true });
	});
	afterAll(() => rmSync(root, { recursive: true, force: true }));

	it("reports its version-independent contract: formats list and a format's targets", async () => {
		const all = await listFileeFormats(undefined, { ...filee, dataDir });
		expect(all).toContain("png");
		const heic = await listFileeFormats("heic", { ...filee, dataDir });
		expect(heic).toContain("converts to");
		await expect(listFileeFormats("not-a-format", { ...filee, dataDir })).rejects.toThrow("unknown format");
	});

	it("converts a Korean-named Markdown file to DOCX in an arbitrary folder", async () => {
		const source = join(root, "문서 폴더", "분기 보고서.md");
		writeFileSync(source, REPORT);
		const result = await convertWithFilee({ inputs: [source], to: "docx" }, { ...filee, dataDir });
		expect(result).toMatchObject({ converted: 1, failed: 0 });
		const output = result.files[0]?.outputs[0] ?? "";
		expect(output).toBe(join(root, "문서 폴더", "분기 보고서.docx"));
		expect(statSync(output).size).toBeGreaterThan(1000);
	});

	it("numbers an existing output by default and replaces it with overwrite", async () => {
		const source = join(root, "문서 폴더", "분기 보고서.md");
		const numbered = await convertWithFilee({ inputs: [source], to: "docx" }, { ...filee, dataDir });
		const second = numbered.files[0]?.outputs[0] ?? "";
		expect(second).not.toBe(join(root, "문서 폴더", "분기 보고서.docx"));
		expect(existsSync(second)).toBe(true);

		const target = join(root, "문서 폴더", "분기 보고서.docx");
		writeFileSync(target, "SENTINEL");
		const replaced = await convertWithFilee({ inputs: [source], to: "docx", overwrite: true }, { ...filee, dataDir });
		expect(replaced.files[0]?.outputs[0]).toBe(target);
		expect(readFileSync(target).subarray(0, 2).toString()).toBe("PK");

		writeFileSync(target, "SENTINEL");
		const skipped = await convertWithFilee(
			{ inputs: [source], to: "docx", skipExisting: true },
			{ ...filee, dataDir },
		);
		expect(skipped.skipped).toBe(1);
		expect(readFileSync(target, "utf8")).toBe("SENTINEL");
	});

	it("surfaces per-file failures without throwing, and bad targets as errors", async () => {
		const bad = join(root, "broken.png");
		writeFileSync(bad, "not an image");
		const result = await convertWithFilee({ inputs: [bad], to: "jpg" }, { ...filee, dataDir });
		expect(result.failed).toBe(1);
		expect(result.files[0]?.state).toBe("failed");
		await expect(convertWithFilee({ inputs: [bad], to: "no-such-format" }, { ...filee, dataDir })).rejects.toThrow(
			"unknown format",
		);
		await expect(
			convertWithFilee({ inputs: [join(root, "missing.png")], to: "jpg" }, { ...filee, dataDir }),
		).rejects.toThrow("not found");
	});

	it("registers converted copies in dupey families the way dupey scan groups them", async () => {
		const dir = join(root, "family");
		mkdirSync(dir);
		const source = join(dir, "report.docx");
		const seed = await convertWithFilee(
			{ inputs: [join(root, "문서 폴더", "분기 보고서.md")], to: "docx", outputDir: dir },
			{ ...filee, dataDir },
		);
		const seeded = seed.files[0]?.outputs[0] ?? "";
		writeFileSync(source, readFileSync(seeded));
		rmSync(seeded, { force: true });
		utimesSync(source, 1_700_000_000, 1_700_000_000);

		const out = await convertAndRelate(
			{ inputs: [source], to: "hwpx", keepDates: true },
			{ filee: { ...filee, dataDir }, dupey: {} },
		);
		const hwpx = out.links.find((link) => link.output.endsWith("report.hwpx"));
		expect(hwpx?.dupey.relation).toBe("exact");

		const asTxt = await convertAndRelate(
			{ inputs: [source], to: "txt" },
			{ filee: { ...filee, dataDir }, dupey: {} },
		);
		expect(asTxt.links[0]?.dupey.relation).toBe("near");

		const asHtml = await convertAndRelate(
			{ inputs: [source], to: "html" },
			{ filee: { ...filee, dataDir }, dupey: {} },
		);
		expect(asHtml.links[0]?.dupey.relation).toBe("unsupported");

		// The verdicts must agree with what a real directory scan builds.
		const scan = await scanWithDupey(dir);
		const family = scan.families.find((candidate) => candidate.files.some((file) => file.endsWith("report.docx")));
		expect(family).toBeDefined();
		const members = (family?.files ?? []).map((file) => file.split("/").pop());
		expect(members).toEqual(expect.arrayContaining(["report.docx", "report.hwpx", "report.txt"]));
		expect(members).not.toContain("report.html");
		const edge = family?.edges.find(
			(candidate) =>
				`${candidate.a}${candidate.b}`.includes("report.hwpx") &&
				`${candidate.a}${candidate.b}`.includes("report.docx"),
		);
		expect(edge?.relation).toBe("exact");
	});
});
