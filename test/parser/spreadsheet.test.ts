import { describe, expect, it } from "vitest";
import type * as XLSX from "xlsx";
import { createDefaultParserRegistry } from "../../src/parser/defaults.ts";
import { ParseError } from "../../src/parser/errors.ts";
import { SpreadsheetParser } from "../../src/parser/spreadsheet.ts";
import { estimateLineTokens, MAX_LINE_TOKENS } from "../../src/parser/spreadsheet-render.ts";
import {
	createBudgetWorkbook,
	createDetailWorkbook,
	createFormWorkbook,
	createLongWorkbook,
	createNumberFormatWorkbook,
	createSideNoteWorkbook,
	createStackedWorkbook,
	createUncachedFormulaWorkbook,
	createWideWorkbook,
	type SpreadsheetBookType,
	writeWorkbook,
} from "../fixtures/spreadsheets.ts";

async function parse(
	workbook: XLSX.WorkBook,
	bookType: SpreadsheetBookType = "xlsx",
	parser = new SpreadsheetParser(),
) {
	return parser.parse({ virtualPath: `/docs/book.${bookType}`, bytes: writeWorkbook(workbook, bookType) });
}

function lines(markdown: string): string[] {
	return markdown.split("\n");
}

function lineContaining(markdown: string, needle: string): string {
	const found = lines(markdown).filter((line) => line.includes(needle));
	expect(found, needle).toHaveLength(1);
	return found[0] as string;
}

describe("SpreadsheetParser routing", () => {
	it("owns the Excel and OpenDocument workbook extensions in the default registry", () => {
		const registry = createDefaultParserRegistry();
		for (const path of [
			"/docs/a.xlsx",
			"/docs/a.xlsm",
			"/docs/a.xlsb",
			"/docs/a.xls",
			"/docs/a.ods",
			"/docs/A.XLSX",
		]) {
			expect(registry.getForVirtualPath(path)?.name, path).toBe("spreadsheet");
		}
	});

	it("reports the workbook format it parsed", async () => {
		const parsed = await parse(createStackedWorkbook(), "xls");
		expect(parsed.metadata).toEqual({ parser: "spreadsheet", format: "xls" });
	});
});

describe("SpreadsheetParser keeps every value next to its labels", () => {
	it("binds values to the real header row below a merged title and an empty row", async () => {
		const { markdown } = await parse(createDetailWorkbook());

		expect(lineContaining(markdown, "외부전문가활용비")).toBe(
			"[세부예산!A354:F354 · 2026년 세부 예산 편성 내역 (단위: 원)] 구분: 직접비 | 비목: 외부전문가활용비 | 세목: 자문료 | 산출근거: 자문 3회 × 105,000 | 편성액: 315000",
		);
		expect(lines(markdown).filter((line) => line.includes("| 편성액: "))).toHaveLength(400);
		expect(markdown).not.toContain("편성액: 편성액");
	});

	it("renders cached formula results with their number formats and repeats merged row keys", async () => {
		const { markdown } = await parse(createBudgetWorkbook());
		const caption = "2026년 연구개발비 편성표 (단위: 천원)";

		expect(lineContaining(markdown, "비목: 인건비")).toBe(
			`[예산총괄!A4:F4 · ${caption}] 구분: 직접비 (comment: 내부 인력 3명 기준) | 비목: 인건비 | 산출근거: 책임 1명 × 12월 / 참여 2명 × 12월 | 편성액: 742,000 | 비율(%): 32.1% | 비고: 내부`,
		);
		expect(lineContaining(markdown, "비목: 연구장비비")).toBe(
			`[예산총괄!A5:F5 · ${caption}] 구분: 직접비 | 비목: 연구장비비 | 산출근거: GPU 서버 1식 | 편성액: 1,200,000`,
		);
		expect(lineContaining(markdown, "구분: 합계")).toBe(
			`[예산총괄!A8:F8 · ${caption}] 구분: 합계 | 편성액: 2,310,000`,
		);
	});

	it("names columns under a two-level merged header", async () => {
		const { markdown } = await parse(createBudgetWorkbook());
		const row = lineContaining(markdown, "비목: 연구수당");

		expect(row.startsWith("[월별집행계획!A3:M3] 비목: 연구수당 | 상반기 > 1월: 61833 | ")).toBe(true);
		expect(row.endsWith(" | 하반기 > 12월: 61837")).toBe(true);
	});

	it("splits tables stacked on one sheet and labels each with its own title and header", async () => {
		const { markdown } = await parse(createStackedWorkbook());

		expect(lineContaining(markdown, "부산")).toBe(
			"[출장!A3:C3 · 국내출장] 일자: 2026-03-02 | 출장지: 부산 | 금액: 120000",
		);
		expect(lineContaining(markdown, "일본")).toBe(
			"[출장!A9:D9 · 해외출장] 일자: 2026-05-20 | 국가: 일본 | 항공료: 450000 | 체재비: 630000",
		);
	});

	it("keeps form-style label/value rows as written instead of inventing column labels", async () => {
		const { markdown } = await parse(createFormWorkbook());

		expect(lineContaining(markdown, "연구책임자")).toBe(
			"[신청서!A3:D3 · 연구개발과제 신청서] 주관기관 | 마커 | 연구책임자 | 홍길동",
		);
		expect(lineContaining(markdown, "총 연구비")).toBe(
			"[신청서!A4:D4 · 연구개발과제 신청서] 총 연구기간 | 2026.01.01~2026.12.31 | 총 연구비 | 2,310,000",
		);
		expect(markdown).not.toContain("과제명: 주관기관");
		expect(lineContaining(markdown, "김철수")).toBe(
			"[신청서!A8:D8 · 참여연구원] 성명: 김철수 | 소속: 마커 | 역할: 공동연구원 | 참여율: 50%",
		);
	});

	it("splits a row longer than a retrieval chunk and repeats the row key on every piece", async () => {
		const { markdown } = await parse(createWideWorkbook());
		const pieces = lines(markdown).filter((line) => line.includes("구분: 인건비"));

		expect(pieces.length).toBeGreaterThan(1);
		for (const line of lines(markdown))
			expect(estimateLineTokens(line), line.slice(0, 40)).toBeLessThanOrEqual(MAX_LINE_TOKENS);
		expect(lineContaining(markdown, "세부사업237 집행액: 4321987")).toContain("구분: 인건비 | ");
		expect(pieces.join(" ").match(/세부사업\d{3} 집행액: /g)).toHaveLength(250);
	});
});

describe("SpreadsheetParser workbook structure and values", () => {
	it("renders each sheet under its own heading, marks hidden sheets, and keeps small key/value sheets as rows", async () => {
		const { markdown } = await parse(createBudgetWorkbook());

		expect(lines(markdown).filter((line) => line.startsWith("## "))).toEqual([
			"## 개요",
			"## 예산총괄",
			"## 월별집행계획",
			"## 인원별상세",
			"## Lookup (hidden)",
		]);
		expect(lineContaining(markdown, "수행기관")).toBe("[개요!A3:C3] 수행기관 | 마커 | 주관");
		expect(lineContaining(markdown, "직원30")).toContain("| 항목18: 987654 | ");
	});

	it("shows a note merged across rows and columns once instead of repeating it on every row", async () => {
		const { markdown } = await parse(createSideNoteWorkbook());

		expect(lineContaining(markdown, "01-01")).toBe(
			"[거래내역!A2:H2] 일련번호: 01-01 | 거래일시: 2026-03-03 | 거래 내용: 회비 입금 | 입금액: 20000 | 작성방법: <일련번호 작성방법> 사업번호-거래순서",
		);
		expect(lineContaining(markdown, "01-02")).toBe(
			"[거래내역!A3:H3] 일련번호: 01-02 | 거래일시: 2026-03-04 | 거래 내용: 회비 입금 | 입금액: 20000",
		);
		expect(markdown.match(/작성방법/g)).toHaveLength(2);
	});

	it("keeps scaled, locale and general number formats honest", async () => {
		const { markdown } = await parse(createNumberFormatWorkbook());
		const values = lines(markdown)
			.filter((line) => line.startsWith("[서식!"))
			.map((line) => line.replace(/^\[[^\]]+\] 값: /, "").replace(/ \| 서식: .*$/, ""));

		expect(values).toEqual([
			"742,000원",
			"742천원 (742000)",
			"4백만원 (3500000)",
			"-742,000",
			"32.1%",
			"2026-07-01",
			"1234",
			"123456789012",
			"0.3",
		]);
	});

	it("shows the formula when a formula cell has no saved result", async () => {
		const { markdown } = await parse(createUncachedFormulaWorkbook());

		expect(lineContaining(markdown, "소계")).toBe("[무캐시!A4:B4] 비목: 소계 | 편성액: =SUM(B2:B3)");
	});

	it.each(["xlsm", "xlsb", "xls", "ods"] as const)(
		"reads %s workbooks with the same row rendering",
		async (bookType) => {
			const { markdown } = await parse(createBudgetWorkbook(), bookType);

			const row = lineContaining(markdown, "비목: 인건비");
			expect(row).toContain("[예산총괄!A4:F4 · 2026년 연구개발비 편성표 (단위: 천원)] 구분: 직접비");
			// SheetJS's ODS writer saves the 0.0% style with two decimals, so that fixture file itself says 32.12%.
			expect(row).toMatch(/\| 편성액: 742,000 \| 비율\(%\): 32\.12?% \| 비고: 내부$/);
			expect(lineContaining(markdown, "비목: 연구장비비")).toContain("구분: 직접비 | 비목: 연구장비비");
		},
	);

	it("indexes the first rows of an oversized sheet and says how many it kept", async () => {
		const parsed = await parse(createLongWorkbook(120), "xlsx", new SpreadsheetParser({ maxRowsPerSheet: 50 }));

		expect(parsed.markdown).toContain("주문번호: ORD-000048 | 금액: 1048");
		expect(parsed.markdown).not.toContain("ORD-000049");
		expect(parsed.diagnostics).toEqual([
			{ code: "parser-warning", severity: "warning", message: 'sheet "주문": indexed the first 50 of 121 rows' },
		]);
	});

	it.each([
		["/docs/a.xlsx", Buffer.from("not a workbook")],
		["/docs/a.ods", Buffer.from("PK\u0003\u0004 truncated zip")],
		["/docs/a.xls", Buffer.from("not xls")],
		["/docs/a.xls", Buffer.from([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1, 0, 0, 0, 0])],
	])("rejects %s bytes that are not a readable workbook with a typed parse error", async (virtualPath, bytes) => {
		await expect(new SpreadsheetParser().parse({ virtualPath, bytes })).rejects.toBeInstanceOf(ParseError);
	});
});
