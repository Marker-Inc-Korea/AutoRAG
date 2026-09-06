import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";
import { resolveVirtualSource, type SourceRoot } from "../filesystem/source-paths.ts";
import { PARSED_FILES_SUBDIR } from "../mirror/paths.ts";
import { redactPII } from "./pii-gate.ts";
import type { PolicyResolution, PolicyTier } from "./policy.ts";
import { wireSourceIdToVirtualPath } from "./wire.ts";

export interface FileShareDiagnostic {
	readonly code: string;
	readonly message: string;
}

export type FileShareResponse =
	| {
			readonly status: "ok";
			readonly fileBase64: string;
			readonly redacted: boolean;
	  }
	| {
			readonly status: "withheld";
			readonly fileBase64: "";
			readonly diagnostic: FileShareDiagnostic;
	  }
	| {
			readonly status: "rejected";
			readonly diagnostic: FileShareDiagnostic;
	  };

export type FileResponse = FileShareResponse;

export interface FileShareOptions {
	readonly resolvePolicy: (virtualPath: string, peerFingerprint?: string) => PolicyResolution;
	readonly workspaceRoots: readonly SourceRoot[];
	readonly parsedMirrorRoot: string;
	readonly maxFileBytes: number;
}

const DENIED_MESSAGE = "The requested file is not available.";
const TOO_LARGE_MESSAGE = "The requested file exceeds the sharing size limit.";
const BINARY_WITHHELD_MESSAGE = "Binary files are not available for this sharing tier.";

const TEXT_EXTENSIONS = new Set([
	".c",
	".cc",
	".cpp",
	".css",
	".csv",
	".eml",
	".go",
	".h",
	".hpp",
	".htm",
	".html",
	".java",
	".js",
	".json",
	".jsx",
	".log",
	".markdown",
	".md",
	".php",
	".py",
	".rb",
	".rs",
	".scss",
	".sh",
	".sql",
	".text",
	".toml",
	".ts",
	".tsx",
	".txt",
	".xml",
	".yaml",
	".yml",
]);

function denied(): FileShareResponse {
	return {
		status: "rejected",
		diagnostic: { code: "policy-denied", message: DENIED_MESSAGE },
	};
}

function isAllowedTier(resolution: PolicyResolution, tier: PolicyTier): boolean {
	return resolution.tier === tier && resolution.allowed === true;
}

function isWithinLimit(size: number, maxFileBytes: number): boolean {
	return Number.isSafeInteger(maxFileBytes) && maxFileBytes >= 0 && size <= maxFileBytes;
}

function tooLarge(): FileShareResponse {
	return {
		status: "rejected",
		diagnostic: { code: "file-too-large", message: TOO_LARGE_MESSAGE },
	};
}

function mirrorPath(parsedRoot: string, virtualPath: string): string {
	const digest = createHash("sha256").update(virtualPath).digest("hex");
	return join(parsedRoot, PARSED_FILES_SUBDIR, `${digest}.md`);
}

function isTextVirtualPath(virtualPath: string): boolean {
	return TEXT_EXTENSIONS.has(extname(virtualPath).toLowerCase());
}

function readOriginalBytes(realPath: string, maxFileBytes: number): Buffer | undefined {
	try {
		const size = statSync(realPath).size;
		if (!isWithinLimit(size, maxFileBytes)) return undefined;
		const bytes = readFileSync(realPath);
		return isWithinLimit(bytes.byteLength, maxFileBytes) ? bytes : undefined;
	} catch {
		return undefined;
	}
}

/**
 * Resolve and serve one peer file request after policy and containment checks.
 *
 * Unknown wire ids, missing files, denied files, malformed virtual paths, and
 * containment failures deliberately share one path-free refusal response so a
 * peer cannot turn this endpoint into an existence oracle.
 */
export function resolveFileShare(
	wireId: string,
	peerFingerprint: string,
	options: FileShareOptions,
): FileShareResponse {
	if (typeof wireId !== "string" || typeof peerFingerprint !== "string") return denied();

	let virtualPath: string | undefined;
	try {
		virtualPath = wireSourceIdToVirtualPath(wireId);
	} catch {
		return denied();
	}
	if (virtualPath === undefined) return denied();

	const resolved = resolveVirtualSource(virtualPath, options.workspaceRoots);
	if (resolved === undefined) return denied();

	let policy: PolicyResolution;
	try {
		policy = options.resolvePolicy(virtualPath, peerFingerprint);
	} catch {
		return denied();
	}

	if (isAllowedTier(policy, "always")) {
		const bytes = readOriginalBytes(resolved.realPath, options.maxFileBytes);
		if (bytes === undefined) {
			try {
				if (statSync(resolved.realPath).size > options.maxFileBytes) return tooLarge();
			} catch {
				// Missing, unreadable, and non-file paths remain indistinguishable from denial.
			}
			return denied();
		}
		return { status: "ok", fileBase64: bytes.toString("base64"), redacted: false };
	}

	if (!isAllowedTier(policy, "peers")) return denied();
	if (!isTextVirtualPath(virtualPath)) {
		return {
			status: "withheld",
			fileBase64: "",
			diagnostic: { code: "policy-denied-binary", message: BINARY_WITHHELD_MESSAGE },
		};
	}

	let markdown: string;
	try {
		markdown = readFileSync(mirrorPath(options.parsedMirrorRoot, virtualPath), "utf8");
	} catch {
		return denied();
	}
	const redacted = redactPII(markdown, { pseudonymize: false }).text;
	const bytes = Buffer.from(redacted, "utf8");
	if (!isWithinLimit(bytes.byteLength, options.maxFileBytes)) return tooLarge();
	return { status: "ok", fileBase64: bytes.toString("base64"), redacted: true };
}
