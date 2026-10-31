import { describe, expect, it } from "vitest";
import { CsvParser } from "../../src/parser/csv.ts";
import { createDefaultParserRegistry } from "../../src/parser/defaults.ts";
import {
	createBudgetCsv,
	createLongCsv,
	createOperatingCostCsv,
	createSemicolonCsv,
	createStackedCsv,
	createStaffTsv,
	createVerbatimCsv,
} from "../fixtures/spreadsheets.ts";

async function parse(virtualPath: string, content: string | Uint8Array, parser = new CsvParser()) {
	return parser.parse({ virtualPath, bytes: typeof content === "string" ? Buffer.from(content, "utf8") : content });
}

function lineContaining(markdown: string, needle: string): string {
	const found = markdown.split("\n").filter((line) => line.includes(needle));
	expect(found, needle).toHaveLength(1);
	return found[0] as string;
}

describe("CsvParser routing", () => {
	it("owns .csv and .tsv in the default registry", () => {
		const registry = createDefaultParserRegistry();
		for (const path of ["/docs/a.csv", "/docs/a.tsv", "/docs/A.CSV"]) {
			expect(registry.getForVirtualPath(path)?.name, path).toBe("csv");
		}
	});

	it("names the file as the sheet and reports the format", async () => {
		const parsed = await parse("/docs/인원현황.tsv", createStaffTsv());

		expect(parsed.markdown.split("\n")[0]).toBe("## 인원현황");
		expect(parsed.metadata).toEqual({ parser: "csv", format: "tsv" });
	});
});

describe("CsvParser keeps every value next to its labels", () => {
	it("renders an Excel CSV export with its caption, header and quoted line break", async () => {
		const { markdown } = await parse("/docs/예산총괄.csv", createBudgetCsv());
		const caption = "2026년 연구개발비 편성표 (단위: 천원)";

		expect(lineContaining(markdown, "비목: 인건비")).toBe(
			`[예산총괄!A4:F4 · ${caption}] 구분: 직접비 | 비목: 인건비 | 산출근거: 책임 1명 × 12월 / 참여 2명 × 12월 | 편성액: 742,000 | 비율(%): 32.1% | 비고: 내부`,
		);
		expect(lineContaining(markdown, "비목: 연구장비비")).toBe(
			`[예산총괄!A5:F5 · ${caption}] 비목: 연구장비비 | 산출근거: GPU 서버 1식 | 편성액: 1,200,000`,
		);
	});

	it("decodes CP949, keeps quoted line breaks and doubled quotes, and tolerates a ragged row", async () => {
		const { markdown } = await parse("/docs/운영비.csv", createOperatingCostCsv());
		const caption = "2026년 운영비 집행계획 (단위: 원)";

		expect(lineContaining(markdown, "회의비")).toBe(
			`[운영비!A4:D4 · ${caption}] 비목: 회의비 | 산출근거: 월 2회 / × 12월 | 편성액: 480,000 | 비고: 내부`,
		);
		expect(lineContaining(markdown, "여비")).toBe(
			`[운영비!A6:D6 · ${caption}] 비목: 여비 "국내" | 산출근거: 국내출장 6회 | 편성액: 1,320,000`,
		);
		expect(lineContaining(markdown, "합계")).toBe(`[운영비!A7:D7 · ${caption}] 비목: 합계`);
	});

	it("reads a BOM-prefixed TSV as tab-separated rows", async () => {
		const { markdown } = await parse("/docs/인원현황.tsv", createStaffTsv());

		expect(lineContaining(markdown, "연구지원팀")).toBe(
			"[인원현황!A2:C2] 부서: 연구지원팀 | 인원: 4 | 예산: 1200000",
		);
		expect(markdown).not.toContain("\uFEFF");
	});

	it.each([
		["with Excel's sep=; line", true],
		["sniffed from the rows", false],
	])("splits semicolon-separated values %s", async (_label, directive) => {
		const { markdown } = await parse("/docs/부품.csv", createSemicolonCsv(directive));

		expect(lineContaining(markdown, "볼트")).toBe("[부품!A2:C2] 품목: 볼트 | 단가: 1,5 | 수량: 100");
		expect(markdown).not.toContain("sep=");
	});

	it("splits tables stacked in one CSV and labels each with its own title and header", async () => {
		const { markdown } = await parse("/docs/출장.csv", createStackedCsv());

		expect(lineContaining(markdown, "일본")).toBe(
			"[출장!A9:D9 · 해외출장] 일자: 2026-05-20 | 국가: 일본 | 항공료: 450000 | 체재비: 630000",
		);
		expect(lineContaining(markdown, "부산")).toBe(
			"[출장!A3:C3 · 국내출장] 일자: 2026-03-02 | 출장지: 부산 | 금액: 120000",
		);
	});

	it("keeps leading zeros, long digit codes, formula-like text and quotes verbatim", async () => {
		const { markdown } = await parse("/docs/코드표.csv", createVerbatimCsv());

		expect(lineContaining(markdown, "00123")).toBe("[코드표!A2:C2] 코드: 00123 | 설명: =1+2 | 수량: 10");
		expect(lineContaining(markdown, "12345678901234567890")).toBe(
			"[코드표!A3:C3] 코드: 12345678901234567890 | 설명: 쉼표, 포함 | 수량: 20",
		);
		expect(lineContaining(markdown, "A-01")).toBe('[코드표!A4:C4] 코드: A-01 | 설명: 따옴표 "인용" | 수량: 30');
	});

	it("indexes the first rows of an oversized file and says how many it kept", async () => {
		const parsed = await parse("/docs/주문.csv", createLongCsv(120), new CsvParser({ maxRows: 50 }));

		expect(parsed.markdown).toContain("주문번호: ORD-000048 | 금액: 1048");
		expect(parsed.markdown).not.toContain("ORD-000049");
		expect(parsed.diagnostics).toEqual([
			{ code: "parser-warning", severity: "warning", message: '"주문.csv": indexed the first 50 of 121 rows' },
		]);
	});
});
