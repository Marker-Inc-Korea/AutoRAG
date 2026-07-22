import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MailExportConnector } from "../../../src/datasource/skills/mail-export/connector.ts";
import { MailExportSkill } from "../../../src/datasource/skills/mail-export/skill.ts";

let tmpRoot: string;

beforeEach(() => {
	tmpRoot = mkdtempSync(join(tmpdir(), "autorag-mail-export-"));
});

afterEach(() => {
	rmSync(tmpRoot, { recursive: true, force: true });
});

const EML = [
	"From: alice@example.com",
	"To: bob@example.com",
	"Subject: Invoice 42",
	"Date: Mon, 11 Mar 2024 10:00:00 +0000",
	"",
	"Please pay invoice 42 by end of month.",
].join("\r\n");

const MBOX = [
	"From alice@example.com Mon Mar 11 10:00:00 2024",
	"From: alice@example.com",
	"Subject: Weekly report",
	"Date: Mon, 11 Mar 2024 10:00:00 +0000",
	"",
	"Numbers are up 5% this week.",
	"From bob@example.com Tue Mar 12 10:00:00 2024",
	"From: bob@example.com",
	"Subject: Re: Weekly report",
	"Date: Tue, 12 Mar 2024 10:00:00 +0000",
	"",
	"Nice work on the numbers.",
].join("\n");

describe("MailExportConnector", () => {
	it("returns not-configured without paths and unavailable when all paths are missing", async () => {
		expect(await new MailExportConnector({}).fetch()).toMatchObject({ ok: false, reason: "not-configured" });
		expect(await new MailExportConnector({ paths: [join(tmpRoot, "nope")] }).fetch()).toMatchObject({
			ok: false,
			reason: "unavailable",
		});
	});

	it("parses eml files and splits mbox archives into messages", async () => {
		const mailDir = join(tmpRoot, "exports");
		mkdirSync(mailDir, { recursive: true });
		writeFileSync(join(mailDir, "invoice.eml"), EML);
		writeFileSync(join(mailDir, "archive.mbox"), MBOX);

		const result = await new MailExportConnector({ paths: [mailDir] }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(3);
			const invoice = result.documents.find((d) => d.docId === "invoice");
			expect(invoice).toMatchObject({ title: "Invoice 42", hierarchy: ["mailboxes", "exports", "invoice"] });
			expect(invoice?.content).toContain("Please pay invoice 42");
			const mboxDocs = result.documents.filter((d) => d.docId.startsWith("archive-"));
			expect(mboxDocs).toHaveLength(2);
			expect(mboxDocs[0]?.content).toContain("Numbers are up 5%");
		}
	});

	it("keeps warnings free of filesystem paths when some paths are unreadable", async () => {
		const mailDir = join(tmpRoot, "exports");
		mkdirSync(mailDir, { recursive: true });
		writeFileSync(join(mailDir, "invoice.eml"), EML);
		const result = await new MailExportConnector({ paths: [mailDir, join(tmpRoot, "missing")] }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.warnings).toEqual(["1 configured path(s) were unreadable"]);
			expect(JSON.stringify(result.warnings)).not.toContain(tmpRoot);
		}
	});

	it("skips oversized files with a count warning", async () => {
		const mailDir = join(tmpRoot, "exports");
		mkdirSync(mailDir, { recursive: true });
		writeFileSync(join(mailDir, "big.eml"), EML);
		const result = await new MailExportConnector({ paths: [mailDir], maxBytesPerFile: 10 }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(0);
			expect(result.warnings?.[0]).toContain("size limit");
		}
	});
});

describe("MailExportSkill", () => {
	it("indexes and searches with opaque /mail-export sources", async () => {
		const mailDir = join(tmpRoot, "exports");
		mkdirSync(mailDir, { recursive: true });
		writeFileSync(join(mailDir, "invoice.eml"), EML);
		const skill = new MailExportSkill({
			instanceId: "archive",
			connectorOptions: { paths: [mailDir] },
		});
		expect(skill.describe()).toMatchObject({ name: "mail-export", datasourceId: "mail-export" });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("invoice pay month", { topK: 5 });
		expect(hits?.[0]?.source).toMatch(/^\/mail-export\/archive\/chunks\//);
		expect(JSON.stringify(hits?.[0]?.source)).not.toContain(tmpRoot);
	});
});
