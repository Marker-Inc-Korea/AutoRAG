import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { planSourceRoots } from "../../src/filesystem/source-paths.ts";
import { parsedMirrorRoot, parsedOutputPath } from "../../src/mirror/paths.ts";
import { type FileShareOptions, type FileShareResponse, resolveFileShare } from "../../src/p2p/file-sharing.ts";
import type { PolicyResolution } from "../../src/p2p/policy.ts";
import { resetWireMapping, wireSourceId } from "../../src/p2p/wire.ts";

let root: string;
let workspace: string;
let sourceRoot: string;
let parsedRoot: string;
let sourceRoots: ReturnType<typeof planSourceRoots>;
const peerFingerprint = "peer-a";

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-p2p-file-sharing-"));
	workspace = join(root, "workspace");
	sourceRoot = join(workspace, "docs");
	parsedRoot = parsedMirrorRoot(workspace);
	mkdirSync(sourceRoot, { recursive: true });
	mkdirSync(join(parsedRoot, "files"), { recursive: true });
	sourceRoots = planSourceRoots([sourceRoot]);
	resetWireMapping();
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function resolution(tier: PolicyResolution["tier"], allowed: boolean): PolicyResolution {
	return {
		tier,
		allowed,
		shareBytes: tier === "always" && allowed,
		redact: tier !== "always",
	};
}

function options(resolvePolicy: FileShareOptions["resolvePolicy"], maxFileBytes = 1024): FileShareOptions {
	return {
		resolvePolicy,
		workspaceRoots: sourceRoots,
		parsedMirrorRoot: parsedRoot,
		maxFileBytes,
	};
}

type SuccessfulFileShare = Extract<FileShareResponse, { status: "ok" }>;

function requireOk(response: FileShareResponse): SuccessfulFileShare {
	if (response.status !== "ok") throw new Error(`expected successful file share, got ${response.status}`);
	return response;
}

function decode(response: SuccessfulFileShare): string {
	return Buffer.from(response.fileBase64, "base64").toString("utf8");
}

function mirror(virtualPath: string, markdown: string): void {
	const outputPath = parsedOutputPath(workspace, virtualPath);
	writeFileSync(outputPath, markdown);
}

describe("resolveFileShare", () => {
	it("returns verbatim original bytes for an always-tier text file", () => {
		const virtualPath = "/docs/readme.txt";
		const original = "Original bytes, including alice@example.com\\n";
		writeFileSync(join(sourceRoot, "readme.txt"), original);
		const response = requireOk(
			resolveFileShare(
				wireSourceId(virtualPath),
				peerFingerprint,
				options((source, peer) => {
					expect(source).toBe(virtualPath);
					expect(peer).toBe(peerFingerprint);
					return resolution("always", true);
				}),
			),
		);

		expect(decode(response)).toBe(original);
		expect(response.redacted).toBe(false);
	});

	it("returns PII-redacted extracted markdown for a peers-tier text file", () => {
		const virtualPath = "/docs/notes.md";
		writeFileSync(join(sourceRoot, "notes.md"), "Original source");
		mirror(virtualPath, "Extracted contact: alice@example.com\\n");
		const response = requireOk(
			resolveFileShare(
				wireSourceId(virtualPath),
				peerFingerprint,
				options((source, peer) => {
					expect(source).toBe(virtualPath);
					expect(peer).toBe(peerFingerprint);
					return resolution("peers", true);
				}),
			),
		);

		expect(decode(response)).toBe("Extracted contact: [EMAIL]\\n");
		expect(response.redacted).toBe(true);
	});

	it("withholds peers-tier binary files without returning bytes", () => {
		const virtualPath = "/docs/report.pdf";
		writeFileSync(join(sourceRoot, "report.pdf"), Buffer.from("%PDF-1.7\\nsecret"));
		mirror(virtualPath, "Extracted PDF text");
		const response = resolveFileShare(
			wireSourceId(virtualPath),
			peerFingerprint,
			options(() => resolution("peers", true)),
		);

		expect(response).toEqual({
			status: "withheld",
			fileBase64: "",
			diagnostic: { code: "policy-denied-binary", message: expect.any(String) },
		});
	});

	it("uses the same refusal shape for denied and unknown wire ids", () => {
		const deniedPath = "/docs/private.txt";
		writeFileSync(join(sourceRoot, "private.txt"), "private");
		const resolvePolicy = (source: string) => {
			expect(source).toBe(deniedPath);
			return resolution("never", false);
		};
		const denied = resolveFileShare(wireSourceId(deniedPath), peerFingerprint, options(resolvePolicy));
		const nonexistent = resolveFileShare(wireSourceId("/docs/missing.txt"), peerFingerprint, options(resolvePolicy));
		const unknown = resolveFileShare("/docs/unknown-wire-id", peerFingerprint, options(resolvePolicy));

		expect(denied).toEqual(nonexistent);
		expect(denied).toEqual(unknown);
		expect(denied).toEqual({
			status: "rejected",
			diagnostic: { code: "policy-denied", message: expect.any(String) },
		});
	});

	it("denies a symlink that resolves outside the configured source root", () => {
		const virtualPath = "/docs/escape.txt";
		const outsidePath = join(workspace, "outside.txt");
		writeFileSync(outsidePath, "outside secret");
		symlinkSync(outsidePath, join(sourceRoot, "escape.txt"));
		const response = resolveFileShare(
			wireSourceId(virtualPath),
			peerFingerprint,
			options(() => resolution("always", true)),
		);

		expect(response).toEqual({
			status: "rejected",
			diagnostic: { code: "policy-denied", message: expect.any(String) },
		});
	});

	it("refuses an always-tier file larger than maxFileBytes", () => {
		const virtualPath = "/docs/large.txt";
		writeFileSync(join(sourceRoot, "large.txt"), "0123456789");
		const response = resolveFileShare(
			wireSourceId(virtualPath),
			peerFingerprint,
			options(() => resolution("always", true), 5),
		);

		expect(response).toEqual({
			status: "rejected",
			diagnostic: { code: "file-too-large", message: expect.any(String) },
		});
	});
});
