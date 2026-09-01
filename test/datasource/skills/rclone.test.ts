import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { RcloneConnector, type RcloneRunResult } from "../../../src/datasource/skills/cloud-drive/rclone-connector.ts";
import { CloudDriveSkill } from "../../../src/datasource/skills/cloud-drive/skill.ts";

const LISTING = JSON.stringify([
	{
		Path: "contracts/vendor.txt",
		Name: "vendor.txt",
		Size: 120,
		Hashes: { md5: "vendor-v1" },
		ID: "drive-file-1",
		MimeType: "text/plain",
		ModTime: "2026-05-20T00:00:00.000Z",
	},
	{ Path: "logo.png", Name: "logo.png", Size: 5000, MimeType: "image/png", ModTime: "2026-05-21T00:00:00.000Z" },
	{ Path: "notes.md", Name: "notes.md", Size: 80, MimeType: "text/markdown", ModTime: "2026-05-22T00:00:00.000Z" },
]);

function runnerFrom(handler: (args: readonly string[]) => RcloneRunResult) {
	return async (args: readonly string[]): Promise<RcloneRunResult> => handler(args);
}

const ok = (stdout: string): RcloneRunResult => ({ ok: true, stdout, stderr: "", code: 0 });

const roots: string[] = [];

afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function workspace(): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-rclone-test-"));
	roots.push(root);
	return root;
}

describe("RcloneConnector", () => {
	it("returns not-configured without a remote", async () => {
		expect(await new RcloneConnector({}).fetch()).toMatchObject({ ok: false, reason: "not-configured" });
	});

	it("runs rclone directly without injecting a managed config", async () => {
		const root = workspace();
		const binary = join(root, "rclone");
		const log = join(root, "rclone-env.json");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(log)}, JSON.stringify({
  config: process.env.RCLONE_CONFIG,
  cwd: process.cwd(),
}));
process.stdout.write("[]");
`,
		);
		chmodSync(binary, 0o755);
		const result = await new RcloneConnector({
			binaryPath: binary,
			remote: "drive:",
			workspaceRoot: root,
		}).fetch();
		if (!result.ok) throw new Error(`rclone failed: ${JSON.stringify(result)}`);
		expect(result.ok).toBe(true);
		expect(JSON.parse(readFileSync(log, "utf8")).config).toBeUndefined();
	});

	it("passes an explicit operator config through RCLONE_CONFIG", async () => {
		const root = workspace();
		const binary = join(root, "rclone");
		const log = join(root, "rclone-config.json");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(log)}, JSON.stringify({ config: process.env.RCLONE_CONFIG }));
process.stdout.write("[]");
`,
		);
		chmodSync(binary, 0o755);
		const configPath = join(root, "operator-rclone.conf");
		const result = await new RcloneConnector({ binaryPath: binary, remote: "drive:", configPath }).fetch();
		expect(result.ok).toBe(true);
		expect(JSON.parse(readFileSync(log, "utf8")).config).toBe(configPath);
	});

	it("inventories recursively, mirrors indexable files, and preserves folder hierarchy", async () => {
		const copied: string[] = [];
		const root = workspace();
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(LISTING);
			if (args[0] === "copyto") {
				copied.push(args[1] ?? "");
				writeFileSync(
					args[2] ?? "",
					args[1]?.includes("vendor") ? "Contract renews annually with 60-day notice." : "Meeting notes.",
				);
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const result = await new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		}).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents.map((d) => d.docId).sort()).toEqual(["contracts/vendor.txt", "notes.md"]);
			const vendor = result.documents.find((d) => d.docId === "contracts/vendor.txt");
			expect(vendor).toMatchObject({ title: "vendor.txt", hierarchy: ["files", "contracts"] });
			expect(vendor?.content).toContain("renews annually");
			expect(vendor?.metadata).toMatchObject({
				virtualPath: "/cloud-drive/drive-1/files/contracts/vendor.txt",
				remoteId: "drive-file-1",
				hashes: { md5: "vendor-v1" },
			});
			// PNG skipped, reported as count-only warning.
			expect(result.warnings?.some((w) => w.includes("skipped"))).toBe(true);
		}
		expect(copied).toEqual(["gdrive:contracts/vendor.txt", "gdrive:notes.md"]);
	});

	it("downloads zero bodies and reports unchanged on a no-op second sync", async () => {
		const root = workspace();
		const copied: string[] = [];
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(LISTING);
			if (args[0] === "copyto") {
				copied.push(args[1] ?? "");
				writeFileSync(args[2] ?? "", args[1]?.includes("vendor") ? "Vendor body" : "Notes body");
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const connector = new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		});

		const first = await connector.fetch();
		const firstCopyCount = copied.length;
		const second = await connector.fetch();

		expect(first).toMatchObject({ ok: true, changed: true });
		expect(second).toMatchObject({ ok: true, changed: false });
		expect(copied).toHaveLength(firstCopyCount);
	});

	it("persists only the rclone remote name, not a private subpath", async () => {
		const root = workspace();
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(LISTING);
			if (args[0] === "copyto") {
				writeFileSync(args[2] ?? "", "document body");
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		await new RcloneConnector({
			remote: "gdrive:Private/Legal",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		}).fetch();
		const manifest = readFileSync(
			join(root, ".autorag", "datasources", "cloud-drive", "drive-1", "manifest.json"),
			"utf8",
		);
		expect(manifest).toContain('"remoteName":"gdrive"');
		expect(manifest).not.toContain("Private/Legal");
	});

	it("downloads only changed files and removes only deleted mirror entries", async () => {
		const root = workspace();
		let listing = JSON.parse(LISTING) as Array<Record<string, unknown>>;
		const copied: string[] = [];
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(JSON.stringify(listing));
			if (args[0] === "copyto") {
				copied.push(args[1] ?? "");
				writeFileSync(args[2] ?? "", `body:${args[1]}:${copied.length}`);
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const connector = new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		});
		await connector.fetch();
		const mirrorRoot = join(root, ".autorag", "datasources", "cloud-drive", "drive-1", "mirror");
		expect(existsSync(join(mirrorRoot, "contracts", "vendor.txt"))).toBe(true);

		listing = [
			{
				...listing[0],
				Hashes: { md5: "vendor-v2" },
				ModTime: "2026-05-23T00:00:00.000Z",
			},
		];
		copied.length = 0;
		const changed = await connector.fetch();

		expect(changed).toMatchObject({ ok: true, changed: true });
		if (changed.ok) {
			expect(changed.documents.map((document) => document.docId)).toEqual(["contracts/vendor.txt"]);
			expect(changed.deletedDocIds).toEqual(["notes.md"]);
		}
		expect(copied).toEqual(["gdrive:contracts/vendor.txt"]);
		expect(existsSync(join(mirrorRoot, "notes.md"))).toBe(false);
		expect(readFileSync(join(mirrorRoot, "contracts", "vendor.txt"), "utf8")).toContain("vendor.txt");
	});

	it("keeps unchanged chunk content while applying one-file updates", async () => {
		const root = workspace();
		let listing = JSON.parse(LISTING) as Array<Record<string, unknown>>;
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(JSON.stringify(listing));
			if (args[0] === "copyto") {
				writeFileSync(args[2] ?? "", args[1]?.includes("vendor") ? "vendor version one" : "stable notes sentinel");
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const connector = new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		});
		const skill = new CloudDriveSkill({ instanceId: "drive-1", workspaceRoot: root, connector });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		listing = listing.map((entry) =>
			entry.Path === "contracts/vendor.txt"
				? { ...entry, Hashes: { md5: "vendor-v2" }, ModTime: "2026-05-23T00:00:00.000Z" }
				: entry,
		);
		const updatedRunner = runnerFrom((args) => {
			if (args[0] === "lsjson") return ok(JSON.stringify(listing));
			if (args[0] === "copyto") {
				writeFileSync(args[2] ?? "", "vendor version two");
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const updatedSkill = new CloudDriveSkill({
			instanceId: "drive-1",
			workspaceRoot: root,
			connector: new RcloneConnector({
				remote: "gdrive:",
				workspaceRoot: root,
				instanceId: "drive-1",
				runner: updatedRunner,
			}),
		});
		expect(await updatedSkill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const [method] = updatedSkill.retrievalMethods();
		expect((await method?.retrieve("stable notes sentinel", { topK: 3 }))?.[0]?.content).toContain(
			"stable notes sentinel",
		);
		expect((await method?.retrieve("vendor version two", { topK: 3 }))?.[0]?.content).toContain("vendor version two");
	});

	it("keeps the previous completed snapshot when a changed download is interrupted", async () => {
		const root = workspace();
		let version = 1;
		const runner = runnerFrom((args) => {
			if (args[0] === "lsjson") {
				return ok(
					JSON.stringify([
						{
							Path: "contracts/vendor.txt",
							Name: "vendor.txt",
							Size: 120,
							Hashes: { md5: `vendor-v${version}` },
							ModTime: `2026-05-2${version}T00:00:00.000Z`,
						},
					]),
				);
			}
			if (args[0] === "copyto") {
				if (version === 2) return { ok: false, stdout: "", stderr: "interrupted", code: 1 };
				writeFileSync(args[2] ?? "", "stable previous content");
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		});
		const connector = new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		});
		expect(await connector.fetch()).toMatchObject({ ok: true, changed: true });
		version = 2;
		const interrupted = await connector.fetch();

		expect(interrupted).toMatchObject({ ok: false });
		const recovered = new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner: runnerFrom((args) => {
				if (args[0] === "lsjson") {
					return ok(
						JSON.stringify([
							{
								Path: "contracts/vendor.txt",
								Name: "vendor.txt",
								Size: 120,
								Hashes: { md5: "vendor-v2" },
								ModTime: "2026-05-22T00:00:00.000Z",
							},
						]),
					);
				}
				if (args[0] === "copyto") {
					writeFileSync(args[2] ?? "", "updated recovered content");
					return ok("");
				}
				return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
			}),
		});
		const result = await recovered.fetch();
		expect(result).toMatchObject({ ok: true, changed: true });
		if (result.ok) expect(result.documents[0]?.content).toContain("updated recovered");
	});

	it("keeps search available against the previous snapshot during a slow sync", async () => {
		const root = workspace();
		let version = 1;
		let releaseCopy: (() => void) | undefined;
		let copyStarted: (() => void) | undefined;
		const copyStartedSignal = new Promise<void>((resolve) => {
			copyStarted = resolve;
		});
		const runner = async (args: readonly string[]): Promise<RcloneRunResult> => {
			if (args[0] === "lsjson") {
				return ok(
					JSON.stringify([
						{
							Path: "status.txt",
							Name: "status.txt",
							Size: 50,
							Hashes: { md5: `status-v${version}` },
							ModTime: `2026-05-2${version}T00:00:00.000Z`,
						},
					]),
				);
			}
			if (args[0] === "copyto") {
				if (version === 2) {
					copyStarted?.();
					await new Promise<void>((resolve) => {
						releaseCopy = resolve;
					});
				}
				writeFileSync(args[2] ?? "", version === 1 ? "previous snapshot alpha" : "new snapshot beta");
				return ok("");
			}
			return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
		};
		const connector = new RcloneConnector({
			remote: "gdrive:",
			workspaceRoot: root,
			instanceId: "drive-1",
			runner,
		});
		const skill = new CloudDriveSkill({ instanceId: "drive-1", workspaceRoot: root, connector });
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
		version = 2;
		const slowSync = skill.index();
		await copyStartedSignal;

		const [method] = skill.retrievalMethods();
		const during = await method?.retrieve("previous snapshot alpha", { topK: 3 });
		expect(during?.[0]?.content).toContain("previous snapshot alpha");

		releaseCopy?.();
		expect(await slowSync).toMatchObject({ ok: true, chunkCount: 1 });
		const after = await method?.retrieve("new snapshot beta", { topK: 3 });
		expect(after?.[0]?.content).toContain("new snapshot beta");
	});

	it("classifies config/auth/permission failures without leaking stderr", async () => {
		const cases: [string, string][] = [
			["didn't find section in config file", "not-configured"],
			["failed to refresh oauth token: 401 unauthorized", "auth"],
			["googleapi: Error 403: forbidden", "permission"],
		];
		for (const [stderr, reason] of cases) {
			const runner = runnerFrom(() => ({ ok: false, stdout: "", stderr, code: 1 }));
			const result = await new RcloneConnector({ remote: "gdrive:", runner }).fetch();
			expect(result).toMatchObject({ ok: false, reason });
			if (!result.ok) {
				expect(result.message).not.toContain("config file");
				expect(result.message).not.toContain("401");
			}
		}
	});

	it("degrades per-file cat failures to a count warning", async () => {
		const runner = runnerFrom((args) =>
			args[0] === "lsjson" ? ok(LISTING) : { ok: false, stdout: "", stderr: "read failed", code: 1 },
		);
		const result = await new RcloneConnector({ remote: "gdrive:", runner }).fetch();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.documents).toHaveLength(0);
			expect(result.warnings?.some((w) => w.includes("failed to read"))).toBe(true);
		}
	});

	it("plugs into CloudDriveSkill for indexing and opaque-source search", async () => {
		const root = workspace();
		const runner = runnerFrom((args) =>
			args[0] === "lsjson"
				? ok(LISTING)
				: (() => {
						writeFileSync(args[2] ?? "", "Contract renews annually with 60-day cancellation notice.");
						return ok("");
					})(),
		);
		const skill = new CloudDriveSkill({
			instanceId: "drive-1",
			workspaceRoot: root,
			connector: new RcloneConnector({ remote: "gdrive:", workspaceRoot: root, instanceId: "drive-1", runner }),
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		const chunksPath = join(root, ".autorag", "datasources", "cloud-drive", "drive-1", "chunks.json");
		const firstMtime = statSync(chunksPath).mtimeMs;
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 2 });
		expect(statSync(chunksPath).mtimeMs).toBe(firstMtime);
		const [method] = skill.retrievalMethods();
		const hits = await method?.retrieve("contract cancellation notice", { topK: 3 });
		expect(hits?.[0]?.source).toMatch(/^\/cloud-drive\/drive-1\/chunks\//);
	});
});
