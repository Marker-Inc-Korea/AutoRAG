import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import type { RcloneRunResult } from "../../../src/datasource/skills/cloud-drive/rclone-connector.ts";
import { CloudDriveSkill } from "../../../src/datasource/skills/cloud-drive/skill.ts";

const roots: string[] = [];
const PARSE_ERROR = "temporary parser failure: malformed cross-reference table";

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function fixture(extensions?: readonly string[]) {
	const root = mkdtempSync(join(tmpdir(), "autorag-rclone-parse-retry-"));
	roots.push(root);
	const base = join(root, ".autorag", "datasources", "cloud-drive", "drive-1");
	const files = new Map<string, string>();
	const failingBodies = new Set<string>();
	const emptyBodies = new Set<string>();
	const missingDownloads = new Set<string>();
	const downloads: string[] = [];
	const parsedBodies: string[] = [];
	const ok = (stdout = ""): RcloneRunResult => ({ ok: true, stdout, stderr: "", code: 0 });
	const createSkill = () =>
		new CloudDriveSkill({
			workspaceRoot: root,
			instanceId: "drive-1",
			connectorOptions: {
				remote: "gdrive:",
				extensions,
				runner: async (args) => {
					if (args[0] === "lsjson") {
						return ok(
							JSON.stringify(
								[...files].map(([path, body]) => ({
									Path: path,
									Name: basename(path),
									Size: Buffer.byteLength(body),
									Hashes: { md5: body },
								})),
							),
						);
					}
					if (args[0] !== "copyto") throw new Error(`unexpected rclone command: ${args[0]}`);
					const path = args[1].slice("gdrive:".length);
					const body = files.get(path);
					if (body === undefined) throw new Error(`missing remote fixture: ${path}`);
					downloads.push(path);
					if (!missingDownloads.has(path)) writeFileSync(args[2], body);
					return ok();
				},
				parserOptions: {
					kordoc: {
						parse: async (bytes) => {
							const body = Buffer.from(bytes).toString("utf8");
							parsedBodies.push(body);
							return failingBodies.has(body)
								? { success: false, code: "PARSE_ERROR", error: PARSE_ERROR }
								: { success: true, markdown: emptyBodies.has(body) ? " \n " : body };
						},
					},
				},
			},
		});
	return {
		files,
		failingBodies,
		emptyBodies,
		missingDownloads,
		downloads,
		parsedBodies,
		createSkill,
		mirror: (path: string) => join(base, "mirror", path),
		manifest: () =>
			JSON.parse(readFileSync(join(base, "manifest.json"), "utf8")) as {
				entries: Array<{ path: string; hashes: { md5: string } }>;
			},
	};
}

async function search(skill: CloudDriveSkill, query: string): Promise<string[]> {
	const [method] = skill.retrievalMethods();
	return (await method.retrieve(query, { topK: 10 })).map((hit) => hit.content);
}

describe("rclone parse retry", () => {
	it("keeps the completed version after a parse failure and retries it across skill instances", async () => {
		const f = fixture();
		f.files.set("contracts/vendor.pdf", "archivedcontract cancellation terms");
		const skill = f.createSkill();
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
		const completedManifest = f.manifest();
		f.files.set("contracts/vendor.pdf", "renewedcontract replacement terms");
		f.failingBodies.add("renewedcontract replacement terms");

		const failed = await skill.index();
		expect(failed).toMatchObject({ ok: true, chunkCount: 1 });
		expect(failed.diagnostics).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ severity: "warning", message: expect.stringContaining(PARSE_ERROR) }),
			]),
		);
		expect(f.manifest()).toEqual(completedManifest);
		expect(readFileSync(f.mirror("contracts/vendor.pdf"), "utf8")).toBe("archivedcontract cancellation terms");
		expect(await search(skill, "archivedcontract")).toEqual(["archivedcontract cancellation terms"]);
		expect(await search(skill, "renewedcontract")).toEqual([]);

		f.failingBodies.clear();
		const recovered = f.createSkill();
		expect(await recovered.index()).toMatchObject({ ok: true, chunkCount: 1 });
		expect(f.downloads).toHaveLength(3);
		expect(f.parsedBodies).toHaveLength(3);
		expect(await search(recovered, "renewedcontract")).toEqual(["renewedcontract replacement terms"]);
		expect(await search(recovered, "archivedcontract")).toEqual([]);
		expect(f.manifest().entries[0].hashes.md5).toBe("renewedcontract replacement terms");
		expect(readFileSync(f.mirror("contracts/vendor.pdf"), "utf8")).toBe("renewedcontract replacement terms");
		expect(await f.createSkill().index()).toMatchObject({ ok: true, chunkCount: 1 });
		expect(f.downloads).toHaveLength(3);
		expect(f.parsedBodies).toHaveLength(3);
	});

	it("advances successful updates and deletions while retaining a failed file's completed version", async () => {
		const f = fixture();
		f.files.set("bad.pdf", "retaineddocument original");
		f.files.set("good.pdf", "oldgooddocument original");
		f.files.set("removed.pdf", "deleteddocument original");
		const skill = f.createSkill();
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 3 });
		f.files.set("bad.pdf", "brokenreplacement unavailable");
		f.failingBodies.add("brokenreplacement unavailable");
		f.files.set("good.pdf", "updateddocument current");
		f.files.delete("removed.pdf");
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		expect(await search(skill, "retaineddocument")).toEqual(["retaineddocument original"]);
		expect(await search(skill, "updateddocument")).toEqual(["updateddocument current"]);
		expect(await search(skill, "oldgooddocument deleteddocument")).toEqual([]);
		expect(f.manifest().entries.map((entry) => [entry.path, entry.hashes.md5])).toEqual([
			["bad.pdf", "retaineddocument original"],
			["good.pdf", "updateddocument current"],
		]);
		expect(readFileSync(f.mirror("bad.pdf"), "utf8")).toBe("retaineddocument original");
		expect(readFileSync(f.mirror("good.pdf"), "utf8")).toBe("updateddocument current");
		expect(existsSync(f.mirror("removed.pdf"))).toBe(false);
	});

	it("does not acknowledge or publish a new file until parsing succeeds", async () => {
		const f = fixture();
		f.files.set("new.pdf", "recovereddocument searchable");
		f.failingBodies.add("recovereddocument searchable");
		expect(await f.createSkill().index()).toMatchObject({ ok: true, chunkCount: 0 });
		expect(f.manifest().entries).toEqual([]);
		expect(existsSync(f.mirror("new.pdf"))).toBe(false);
		expect(existsSync(`${f.mirror("new.pdf")}.autorag-tmp`)).toBe(false);
		f.failingBodies.clear();
		const recovered = f.createSkill();
		expect(await recovered.index()).toMatchObject({ ok: true, chunkCount: 1 });
		expect(await search(recovered, "recovereddocument")).toEqual(["recovereddocument searchable"]);
		expect(f.downloads).toEqual(["new.pdf", "new.pdf"]);
		expect(f.manifest().entries[0].path).toBe("new.pdf");
	});

	it("needs no download when a failed remote update reverts to the completed version", async () => {
		const f = fixture();
		f.files.set("version.pdf", "originaldocument stable");
		const skill = f.createSkill();
		await skill.index();
		const completedManifest = f.manifest();
		f.files.set("version.pdf", "invaliddocument replacement");
		f.failingBodies.add("invaliddocument replacement");
		await skill.index();
		f.files.set("version.pdf", "originaldocument stable");
		const reverted = f.createSkill();
		expect(await reverted.index()).toMatchObject({ ok: true, chunkCount: 1 });
		expect(f.downloads).toHaveLength(2);
		expect(f.parsedBodies).toHaveLength(2);
		expect(f.manifest()).toEqual(completedManifest);
		expect(readFileSync(f.mirror("version.pdf"), "utf8")).toBe("originaldocument stable");
		expect(await search(reverted, "originaldocument")).toEqual(["originaldocument stable"]);
	});

	it("removes previous chunks after a successful empty parse without retrying unchanged content", async () => {
		const f = fixture();
		f.files.set("empty.pdf", "outdateddocument previous content");
		const skill = f.createSkill();
		await skill.index();
		f.files.set("empty.pdf", "emptydocument scanned blank page");
		f.emptyBodies.add("emptydocument scanned blank page");
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 0 });
		expect(await search(skill, "outdateddocument")).toEqual([]);
		expect(f.manifest().entries[0].hashes.md5).toBe("emptydocument scanned blank page");
		expect(readFileSync(f.mirror("empty.pdf"), "utf8")).toBe("emptydocument scanned blank page");
		expect(await f.createSkill().index()).toMatchObject({ ok: true, chunkCount: 0 });
		expect(f.downloads).toHaveLength(2);
		expect(f.parsedBodies).toHaveLength(2);
	});

	it("does not repeatedly download an explicitly allowed extension without a parser", async () => {
		const f = fixture([".unsupported"]);
		f.files.set("archive.unsupported", "opaque document bytes");
		expect(await f.createSkill().index()).toMatchObject({ ok: true, chunkCount: 0 });
		expect(await f.createSkill().index()).toMatchObject({ ok: true, chunkCount: 0 });
		expect(f.downloads).toEqual(["archive.unsupported"]);
		expect(f.parsedBodies).toEqual([]);
	});

	it("preserves completed data and retries when a successful copy leaves no readable staged file", async () => {
		const f = fixture();
		f.files.set("missing.pdf", "retainedsnapshot original bytes");
		const skill = f.createSkill();
		await skill.index();
		const completedManifest = f.manifest();
		f.files.set("missing.pdf", "recoveredsnapshot new bytes");
		f.missingDownloads.add("missing.pdf");
		const failed = await skill.index();
		expect(failed).toMatchObject({ ok: true, chunkCount: 1 });
		expect(failed.diagnostics).toEqual(
			expect.arrayContaining([
				expect.objectContaining({ severity: "warning", message: expect.stringContaining("ENOENT") }),
			]),
		);
		expect(f.manifest()).toEqual(completedManifest);
		expect(readFileSync(f.mirror("missing.pdf"), "utf8")).toBe("retainedsnapshot original bytes");
		expect(await search(skill, "retainedsnapshot")).toEqual(["retainedsnapshot original bytes"]);
		f.missingDownloads.clear();
		const recovered = f.createSkill();
		expect(await recovered.index()).toMatchObject({ ok: true, chunkCount: 1 });
		expect(await search(recovered, "recoveredsnapshot")).toEqual(["recoveredsnapshot new bytes"]);
		expect(f.downloads).toHaveLength(3);
	});
});
