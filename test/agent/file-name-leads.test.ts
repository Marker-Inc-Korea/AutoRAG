import { describe, expect, it } from "vitest";
import { formatFileNameLeads } from "../../src/agent/file-name-leads.ts";

const hit = (path: string, ...candidates: string[]) => ({
	entry: { path, type: "file" as const },
	candidates,
});

describe("formatFileNameLeads", () => {
	it("lists the files with the names that found them, and says they are unverified leads to open first", () => {
		const text = formatFileNameLeads({
			names: ["invoice", "인보이스"],
			hits: [hit("/docs/invoice-2026-03.pdf", "invoice", "인보이스"), hit("/docs/bill.txt", "invoice")],
			total: 2,
		});
		expect(text).toContain("/docs/invoice-2026-03.pdf (found by: invoice, 인보이스)");
		expect(text).toContain("/docs/bill.txt (found by: invoice)");
		expect(text).toMatch(/names only|not their contents/iu);
		expect(text).toMatch(/open .* first|read .* first/iu);
	});

	it("recommends the directories the files sit in, most files first", () => {
		const text = formatFileNameLeads({
			names: ["lease"],
			hits: [
				hit("/docs/a/lease-1.pdf", "lease"),
				hit("/docs/a/lease-2.pdf", "lease"),
				hit("/docs/b/lease-3.pdf", "lease"),
			],
			total: 3,
		});
		const dirs = text.slice(text.indexOf("Directories"));
		expect(dirs.indexOf("/docs/a")).toBeGreaterThan(-1);
		expect(dirs.indexOf("/docs/a")).toBeLessThan(dirs.indexOf("/docs/b"));
		expect(dirs).toContain("/docs/a (2 files)");
		expect(dirs).toContain("/docs/b (1 file)");
	});

	it("handles Windows paths", () => {
		const text = formatFileNameLeads({
			names: ["invoice"],
			hits: [hit("C:\\docs\\invoice.pdf", "invoice")],
			total: 1,
		});
		expect(text).toContain("C:\\docs (1 file)");
	});

	it("caps the list and reports how many more matched", () => {
		const hits = Array.from({ length: 40 }, (_, index) => hit(`/docs/f-${index}.txt`, "f"));
		const text = formatFileNameLeads({ names: ["f"], hits, total: 40 });
		expect(text).toContain("/docs/f-0.txt");
		expect(text).not.toContain("/docs/f-39.txt");
		expect(text).toMatch(/\d+ more/u);
	});

	it("says nothing was found when no name matched, instead of leaving the section out", () => {
		const text = formatFileNameLeads({ names: ["invoice", "인보이스"], hits: [], total: 0 });
		expect(text).toContain("invoice, 인보이스");
		expect(text).toMatch(/no file|nothing/iu);
	});
});
