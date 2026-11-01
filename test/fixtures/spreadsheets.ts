import iconv from "iconv-lite";
import * as XLSX from "xlsx";

/**
 * Workbook fixtures for the spreadsheet parser, shaped like real Korean budget and
 * application files: merged title rows above the header, vertical merges in the row
 * key column, cached formula results with number formats, two-level merged headers,
 * stacked tables, form-style label/value rows, and very wide or long sheets.
 */

const range = (ref: string) => XLSX.utils.decode_range(ref);

export type SpreadsheetBookType = "xlsx" | "xlsm" | "xlsb" | "xls" | "ods";

export function writeWorkbook(workbook: XLSX.WorkBook, bookType: SpreadsheetBookType = "xlsx"): Buffer {
	return Buffer.from(XLSX.write(workbook, { type: "buffer", bookType, bookSST: true }));
}

/** 예산총괄: title merged over A1:F1, empty row 2, header on row 3, 직접비 merged over A4:A6, cached SUM results. */
export function createBudgetSheet(): XLSX.WorkSheet {
	const sheet = XLSX.utils.aoa_to_sheet([
		["2026년 연구개발비 편성표 (단위: 천원)"],
		[],
		["구분", "비목", "산출근거", "편성액", "비율(%)", "비고"],
		["직접비", "인건비", "책임 1명 × 12월\n참여 2명 × 12월", null, null, "내부"],
		[null, "연구장비비", "GPU 서버 1식", 1200000, null, ""],
		[null, "연구활동비", "출장·회의", 158000, null, ""],
		["간접비", "간접비", "직접비의 10%", null, null, ""],
		["합계", null, null, null, null, ""],
	]);
	sheet.D4 = { t: "n", v: 742000, f: "SUM(Q4:R4)", z: "#,##0" };
	sheet.D5 = { t: "n", v: 1200000, z: "#,##0" };
	sheet.D6 = { t: "n", v: 158000, z: "#,##0" };
	sheet.D7 = { t: "n", v: 210000, f: "ROUND(SUM(D4:D6)*0.1,-3)", z: "#,##0" };
	sheet.D8 = { t: "n", v: 2310000, f: "SUM(D4:D7)", z: "#,##0" };
	sheet.E4 = { t: "n", v: 0.3212, f: "D4/$D$8", z: "0.0%" };
	sheet.A4.c = [{ a: "검토자", t: "내부 인력 3명 기준" }];
	sheet["!merges"] = [range("A1:F1"), range("A4:A6"), range("A8:C8")];
	sheet["!ref"] = "A1:F8";
	return sheet;
}

/** 월별집행계획: 비목 merged over A1:A2, 상반기 over B1:G1, 하반기 over H1:M1, months on row 2. */
export function createMonthlySheet(): XLSX.WorkSheet {
	const months = Array.from({ length: 12 }, (_, index) => `${index + 1}월`);
	const sheet = XLSX.utils.aoa_to_sheet([
		["비목", "상반기", null, null, null, null, null, "하반기", null, null, null, null, null],
		[null, ...months],
		["연구수당", ...months.map((_, index) => (index === 11 ? 61837 : 61833))],
		["회의비", ...months.map(() => 13166)],
	]);
	sheet["!merges"] = [range("A1:A2"), range("B1:G1"), range("H1:M1")];
	return sheet;
}

/** 인원별상세: 46 columns x 60 rows; 직원30's 항목18 holds 987654. */
export function createWideStaffSheet(): XLSX.WorkSheet {
	const header = [
		"사번",
		"성명",
		...Array.from({ length: 44 }, (_, index) => `항목${String(index + 1).padStart(2, "0")}`),
	];
	const rows: unknown[][] = [header];
	for (let staff = 0; staff < 60; staff++) {
		rows.push([
			`E${1000 + staff}`,
			`직원${staff}`,
			...Array.from({ length: 44 }, (_, column) => ((staff * 44 + column) * 7) % 100000),
		]);
	}
	rows[31][2 + 17] = 987654;
	return XLSX.utils.aoa_to_sheet(rows);
}

export function createBudgetWorkbook(): XLSX.WorkBook {
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(
		workbook,
		XLSX.utils.aoa_to_sheet([
			["항목", "내용", "비고"],
			["과제명", "AI 문서검색 고도화", ""],
			["수행기관", "마커", "주관"],
		]),
		"개요",
	);
	XLSX.utils.book_append_sheet(workbook, createBudgetSheet(), "예산총괄");
	XLSX.utils.book_append_sheet(workbook, createMonthlySheet(), "월별집행계획");
	XLSX.utils.book_append_sheet(workbook, createWideStaffSheet(), "인원별상세");
	XLSX.utils.book_append_sheet(
		workbook,
		XLSX.utils.aoa_to_sheet([
			["코드", "값"],
			["A", 1],
		]),
		"Lookup",
	);
	workbook.Workbook = { Sheets: [{ Hidden: 0 }, { Hidden: 0 }, { Hidden: 0 }, { Hidden: 0 }, { Hidden: 1 }] };
	return workbook;
}

/** 세부예산: 400 data rows below a merged title (row 1), an empty row 2 and the header (row 3). */
export function createDetailWorkbook(): XLSX.WorkBook {
	const rows: unknown[][] = [
		["2026년 세부 예산 편성 내역 (단위: 원)"],
		[],
		["구분", "비목", "세목", "산출근거", "편성액", "비고"],
	];
	for (let index = 0; index < 400; index++) {
		rows.push([
			index < 200 ? "직접비" : "간접비",
			`비목${index}`,
			`세목${index}`,
			`산출근거 설명 ${index}번 항목`,
			1000 + index,
			"",
		]);
	}
	rows[3 + 350] = ["직접비", "외부전문가활용비", "자문료", "자문 3회 × 105,000", 315000, ""];
	const sheet = XLSX.utils.aoa_to_sheet(rows);
	sheet["!merges"] = [range("A1:F1")];
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, sheet, "세부예산");
	return workbook;
}

/** 신청서: form-style label/value rows, then the titled 참여연구원 table. */
export function createFormWorkbook(): XLSX.WorkBook {
	const sheet = XLSX.utils.aoa_to_sheet([
		["연구개발과제 신청서"],
		["과제명", "AI 문서검색 고도화", "과제번호", "RS-2026-0001"],
		["주관기관", "마커", "연구책임자", "홍길동"],
		["총 연구기간", "2026.01.01~2026.12.31", "총 연구비", 2310000],
		[],
		["참여연구원"],
		["성명", "소속", "역할", "참여율"],
		["김철수", "마커", "공동연구원", 0.5],
		["이영희", "마커", "연구보조원", 1],
		["박민수", "마커", "연구보조원", 0.3],
	]);
	sheet.D4.z = "#,##0";
	for (const ref of ["D8", "D9", "D10"]) sheet[ref].z = "0%";
	sheet["!merges"] = [range("A1:D1"), range("A6:D6")];
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, sheet, "신청서");
	return workbook;
}

/** 출장: two titled tables with different headers, separated by an empty row. */
export function createStackedWorkbook(): XLSX.WorkBook {
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(
		workbook,
		XLSX.utils.aoa_to_sheet([
			["국내출장"],
			["일자", "출장지", "금액"],
			["2026-03-02", "부산", 120000],
			["2026-04-11", "대전", 80000],
			["2026-04-28", "광주", 95000],
			[],
			["해외출장"],
			["일자", "국가", "항공료", "체재비"],
			["2026-05-20", "일본", 450000, 630000],
			["2026-06-03", "미국", 1850000, 2100000],
			["2026-09-14", "독일", 1420000, 1980000],
		]),
		"출장",
	);
	return workbook;
}

/** 무캐시: a SUM formula saved without a cached result (openpyxl/pandas-style output). */
export function createUncachedFormulaWorkbook(): XLSX.WorkBook {
	const sheet = XLSX.utils.aoa_to_sheet([
		["비목", "편성액"],
		["장비비", 300000],
		["재료비", 120000],
		["소계", null],
	]);
	sheet.B4 = { t: "n", f: "SUM(B2:B3)" } as XLSX.CellObject;
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, sheet, "무캐시");
	return workbook;
}

/** 세부사업별: 250 Korean-labelled value columns, so one row is far longer than a retrieval chunk. */
export function createWideWorkbook(): XLSX.WorkBook {
	const rows: unknown[][] = [
		["구분", ...Array.from({ length: 250 }, (_, index) => `세부사업${String(index + 1).padStart(3, "0")} 집행액`)],
	];
	for (const label of ["인건비", "장비비", "재료비"]) {
		rows.push([label, ...Array.from({ length: 250 }, (_, column) => 1000 * (column + 1) + label.length)]);
	}
	rows[1][237] = 4321987;
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet(rows), "세부사업별");
	return workbook;
}

/** 거래내역: a ledger with a side note merged over a block of rows and columns (G2:H6). */
export function createSideNoteWorkbook(): XLSX.WorkBook {
	const sheet = XLSX.utils.aoa_to_sheet([
		["일련번호", "거래일시", "거래 내용", "입금액", "출금액", "", "작성방법"],
		["01-01", "2026-03-03", "회비 입금", 20000, null, null, "<일련번호 작성방법> 사업번호-거래순서"],
		["01-02", "2026-03-04", "회비 입금", 20000],
		["02-01", "2026-03-10", "간식 구입", null, 35000],
		["02-02", "2026-03-11", "장소 대관", null, 120000],
		["03-01", "2026-03-20", "후원금", 500000],
	]);
	sheet["!merges"] = [range("G2:H6")];
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, sheet, "거래내역");
	return workbook;
}

/** 서식: one value per row with a Korean or locale-specific number format; column B names the format. */
export function createNumberFormatWorkbook(): XLSX.WorkBook {
	const cases: readonly (readonly [number, string])[] = [
		[742000, '#,##0"원"'],
		[742000, '#,##0,"천원"'],
		[3500000, '#,##0,,"백만원"'],
		[-742000, '_-* #,##0_-;\\-* #,##0_-;_-* "-"_-;_-@_-'],
		[0.3212, "0.0%"],
		[46204, "yyyy-mm-dd"],
		[1234, "[DBNum1][$-412]General"],
		[123456789012, "General"],
		[0.30000000000000004, "General"],
	];
	const sheet: XLSX.WorkSheet = { A1: { t: "s", v: "값" }, B1: { t: "s", v: "서식" } };
	cases.forEach(([value, format], index) => {
		sheet[`A${index + 2}`] = { t: "n", v: value, z: format };
		sheet[`B${index + 2}`] = { t: "s", v: format };
	});
	sheet["!ref"] = `A1:B${cases.length + 1}`;
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, sheet, "서식");
	return workbook;
}

/** A sheet with more data rows than a small row cap. */
export function createLongWorkbook(rowCount: number): XLSX.WorkBook {
	const rows: unknown[][] = [["주문번호", "금액"]];
	for (let index = 0; index < rowCount; index++) rows.push([`ORD-${String(index).padStart(6, "0")}`, 1000 + index]);
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet(rows), "주문");
	return workbook;
}

/** 예산총괄 exported to CSV: shown values ("742,000", "32.1%") and a quoted in-cell line break. */
export function createBudgetCsv(): string {
	return XLSX.utils.sheet_to_csv(createBudgetSheet());
}

/** 출장 as CSV: two titled tables separated by an empty line. */
export function createStackedCsv(): string {
	return XLSX.utils.sheet_to_csv(createStackedWorkbook().Sheets.출장 as XLSX.WorkSheet);
}

/** 운영비: a CP949 CSV with a quoted line break, doubled quotes and a ragged last row. */
export function createOperatingCostCsv(): Buffer {
	const text = [
		"2026년 운영비 집행계획 (단위: 원),,,",
		",,,",
		"비목,산출근거,편성액,비고",
		'회의비,"월 2회\r\n× 12월","480,000",내부',
		'인쇄비,보고서 50부,"250,000",',
		'"여비 ""국내""",국내출장 6회,"1,320,000",',
		"합계",
	].join("\r\n");
	return iconv.encode(`${text}\r\n`, "cp949");
}

/** 인원현황: a UTF-8 TSV with a byte-order mark. */
export function createStaffTsv(): Buffer {
	const text = "부서\t인원\t예산\n연구지원팀\t4\t1200000\n개발팀\t12\t5400000\n운영팀\t3\t800000\n";
	return Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(text, "utf8")]);
}

/** 부품: semicolon-separated values with decimal commas, with or without Excel's `sep=;` first line. */
export function createSemicolonCsv(directive: boolean): string {
	return `${directive ? "sep=;\r\n" : ""}품목;단가;수량\r\n볼트;1,5;100\r\n너트;0,8;200\r\n와셔;0,2;300\r\n`;
}

/** 코드표: values a CSV reader must keep verbatim (leading zeros, 20-digit codes, formula-like text, quotes). */
export function createVerbatimCsv(): string {
	return '코드,설명,수량\n00123,=1+2,10\n12345678901234567890,"쉼표, 포함",20\nA-01,"따옴표 ""인용""",30\n';
}

export function createLongCsv(rowCount: number): string {
	const lines = ["주문번호,금액"];
	for (let index = 0; index < rowCount; index++) lines.push(`ORD-${String(index).padStart(6, "0")},${1000 + index}`);
	return `${lines.join("\n")}\n`;
}
