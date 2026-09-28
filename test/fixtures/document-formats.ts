import iconv from "iconv-lite";
import JSZip from "jszip";
import * as XLSX from "xlsx";

export async function createDocxFixture(text: string): Promise<Buffer> {
	const zip = new JSZip();
	zip.file(
		"[Content_Types].xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
			<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
			<Default Extension="xml" ContentType="application/xml"/>
			<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
		</Types>`),
	);
	zip.file(
		"_rels/.rels",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
			<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
		</Relationships>`),
	);
	zip.file(
		"word/document.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
			<w:body><w:p><w:r><w:t>${escapeXml(text)}</w:t></w:r></w:p></w:body>
		</w:document>`),
	);
	return Buffer.from(await zip.generateAsync({ type: "uint8array" }));
}

export async function createPptxFixture(text: string): Promise<Buffer> {
	const zip = new JSZip();
	zip.file(
		"[Content_Types].xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
			<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
			<Default Extension="xml" ContentType="application/xml"/>
			<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
		</Types>`),
	);
	zip.file(
		"ppt/slides/slide1.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
			xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
			<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>${escapeXml(text)}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
		</p:sld>`),
	);
	return Buffer.from(await zip.generateAsync({ type: "uint8array" }));
}

export async function createXlsxFixture(text: string): Promise<Buffer> {
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([["Topic", text]]), "Summary");
	return Buffer.from(XLSX.write(workbook, { bookType: "xlsx", type: "buffer" }));
}

export function createXlsFixture(text: string): Buffer {
	const workbook = XLSX.utils.book_new();
	const worksheet = XLSX.utils.aoa_to_sheet([
		["Topic", text],
		["Owner", "AutoRAG"],
	]);
	XLSX.utils.book_append_sheet(workbook, worksheet, "Summary");
	return Buffer.from(XLSX.write(workbook, { bookType: "xls", type: "buffer" }));
}

export function createRichXlsFixture(): Buffer {
	const workbook = XLSX.utils.book_new();
	const summary = XLSX.utils.aoa_to_sheet([
		["Whitespace", "  preserve me  "],
		["Pipes", "left | right"],
		["Newlines", "first line\nsecond line"],
		["Backslash", String.raw`C:\docs\file.xls`],
	]);
	const second = XLSX.utils.aoa_to_sheet([
		["Unicode", "한글 marker"],
		["Number", 42],
		["Boolean", true],
	]);
	XLSX.utils.book_append_sheet(workbook, summary, "Summary");
	XLSX.utils.book_append_sheet(workbook, second, "Details");
	return Buffer.from(XLSX.write(workbook, { bookType: "xls", type: "buffer" }));
}

export async function createHwpxFixture(text: string): Promise<Buffer> {
	const zip = new JSZip();
	zip.file(
		"Contents/section0.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<hp:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
			<hp:p><hp:run><hp:t>${escapeXml(text)}</hp:t></hp:run></hp:p>
		</hp:sec>`),
	);
	return Buffer.from(await zip.generateAsync({ type: "uint8array" }));
}

/**
 * HWPX whose outer 1x1 table cell contains an inner 2x2 table, plus a
 * `Contents/header.xml` part carrying a sentinel that must never reach the body.
 * Flat XML text extraction both loses the nesting and leaks the header token.
 */
export async function createNestedTableHwpxFixture(): Promise<Buffer> {
	const zip = new JSZip();
	zip.file("mimetype", "application/hwp+zip");
	zip.file(
		"Contents/header.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<hh:head xmlns:hh="http://www.hancom.co.kr/hwpml/2011/head" version="1.4">
			<hh:refList><hh:fontfaces><hh:fontface>hwpxHeaderLeakToken</hh:fontface></hh:fontfaces></hh:refList>
		</hh:head>`),
	);
	const innerCell = (marker: string) =>
		`<hp:tc><hp:subList><hp:p><hp:run><hp:t>${marker}</hp:t></hp:run></hp:p></hp:subList></hp:tc>`;
	zip.file(
		"Contents/section0.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<hp:sec xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">
			<hp:p><hp:run><hp:t>bodyBeforeMarker</hp:t></hp:run></hp:p>
			<hp:p><hp:run>
				<hp:tbl rowCnt="1" colCnt="1">
					<hp:tr><hp:tc><hp:subList>
						<hp:p><hp:run><hp:t>outerCellMarker</hp:t></hp:run></hp:p>
						<hp:p><hp:run>
							<hp:tbl rowCnt="2" colCnt="2">
								<hp:tr>${innerCell("innerA1Marker")}${innerCell("innerB1Marker")}</hp:tr>
								<hp:tr>${innerCell("innerA2Marker")}${innerCell("innerB2Marker")}</hp:tr>
							</hp:tbl>
						</hp:run></hp:p>
					</hp:subList></hp:tc></hp:tr>
				</hp:tbl>
			</hp:run></hp:p>
			<hp:p><hp:run><hp:t>bodyAfterMarker</hp:t></hp:run></hp:p>
		</hp:sec>`),
	);
	return Buffer.from(await zip.generateAsync({ type: "uint8array" }));
}

/** Real multi-sheet XLSX workbook: sheet boundaries and rows must survive parsing. */
export function createMultiSheetXlsxFixture(): Buffer {
	const workbook = XLSX.utils.book_new();
	XLSX.utils.book_append_sheet(
		workbook,
		XLSX.utils.aoa_to_sheet([
			["Quarter", "Revenue", "Owner"],
			["Q3", 742000, "finance team"],
		]),
		"RevenueSheet",
	);
	XLSX.utils.book_append_sheet(
		workbook,
		XLSX.utils.aoa_to_sheet([
			["Risk", "Status"],
			["supply chain", "open"],
		]),
		"RiskSheet",
	);
	return Buffer.from(XLSX.write(workbook, { bookType: "xlsx", type: "buffer" }));
}

export function createEmlFixture(text: string): Buffer {
	return Buffer.from(
		[
			"From: library@example.com",
			"To: reader@example.com",
			"Subject: AutoRAG parser mail",
			"Content-Type: text/plain; charset=utf-8",
			"",
			text,
			"",
		].join("\r\n"),
		"utf8",
	);
}

export function createEucKrEmlFixture(text: string): Buffer {
	return Buffer.from(
		[
			"From: library@example.com",
			"To: reader@example.com",
			"Subject: =?EUC-KR?B?xde9usau?=",
			"Content-Type: text/plain; charset=euc-kr",
			"Content-Transfer-Encoding: base64",
			"",
			iconv.encode(text, "euc-kr").toString("base64"),
			"",
		].join("\r\n"),
		"ascii",
	);
}

function xml(value: string): string {
	return value.replaceAll(/\t+/g, "").trim();
}

function escapeXml(value: string): string {
	return value
		.replaceAll("&", "&amp;")
		.replaceAll("<", "&lt;")
		.replaceAll(">", "&gt;")
		.replaceAll('"', "&quot;")
		.replaceAll("'", "&apos;");
}
