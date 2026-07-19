function escapePdfText(text: string): string {
	return text.replaceAll("\\", "\\\\").replaceAll("(", "\\(").replaceAll(")", "\\)");
}

export function createMinimalPdfBuffer(text: string): Buffer {
	const textCommand = `BT /F1 12 Tf 72 720 Td (${escapePdfText(text)}) Tj ET`;
	const objects = [
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
		`<< /Length ${Buffer.byteLength(textCommand, "ascii")} >>\nstream\n${textCommand}\nendstream`,
	] as const;

	let pdf = "%PDF-1.4\n";
	const offsets = [0];
	for (const [index, object] of objects.entries()) {
		offsets.push(Buffer.byteLength(pdf, "ascii"));
		pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
	}

	const xrefOffset = Buffer.byteLength(pdf, "ascii");
	pdf += `xref\n0 ${objects.length + 1}\n`;
	pdf += "0000000000 65535 f \n";
	for (const offset of offsets.slice(1)) {
		pdf += `${offset.toString().padStart(10, "0")} 00000 n \n`;
	}
	pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;

	return Buffer.from(pdf, "ascii");
}
