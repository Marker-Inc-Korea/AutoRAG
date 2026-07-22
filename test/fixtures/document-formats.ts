import iconv from "iconv-lite";
import JSZip from "jszip";

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
	const zip = new JSZip();
	zip.file(
		"xl/sharedStrings.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
			<si><t>Topic</t></si>
			<si><t>${escapeXml(text)}</t></si>
		</sst>`),
	);
	zip.file(
		"xl/worksheets/sheet1.xml",
		xml(`<?xml version="1.0" encoding="UTF-8"?>
		<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
			<sheetData><row><c t="s"><v>1</v></c></row></sheetData>
		</worksheet>`),
	);
	return Buffer.from(await zip.generateAsync({ type: "uint8array" }));
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
