/**
 * Live E2E: the pinned filee-cli release is really downloaded, SHA-256
 * verified, unpacked under an isolated AutoRAG home, and converts a file.
 *
 * Gated on AUTORAG_FILEE_INSTALL_LIVE=1 (it downloads ~85 MB from GitHub):
 *
 *   AUTORAG_FILEE_INSTALL_LIVE=1 bun run vitest run test/live-e2e/filee-install.test.ts
 *
 * The test never touches the real ~/.autorag: the home is a temp folder and
 * PATH is emptied so a Homebrew filee-cli cannot answer for the installer.
 */
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
	convertWithFilee,
	ensureFileeBinary,
	FILEE_CLI_VERSION,
	listFileeFormats,
	resolveFileeBinary,
	resolveFileeOptions,
} from "../../src/convert/index.ts";

const LIVE = process.env.AUTORAG_FILEE_INSTALL_LIVE === "1";

describe.skipIf(!LIVE)("live: filee-cli installs itself from the pinned release", () => {
	let root: string;
	let home: string;
	beforeAll(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-filee-install-"));
		home = join(root, "home");
	});
	afterAll(() => rmSync(root, { recursive: true, force: true }));

	it("downloads, verifies and unpacks the real archive, then reuses it", async () => {
		const first = await ensureFileeBinary({ home, ...nodeTestOverride() });
		expect(first.source).toBe("installed");
		expect(first.binaryPath).toBe(join(home, "tools", "filee-cli", FILEE_CLI_VERSION, "filee-cli"));
		expect(statSync(first.binaryPath).size).toBeGreaterThan(1000);
		expect(existsSync(join(home, "tools", "filee-cli", FILEE_CLI_VERSION, "engines"))).toBe(true);
		const second = await ensureFileeBinary({ home });
		expect(second.source).toBe("cached");
	}, 600_000);

	it("resolves the managed copy when PATH has nothing, and it really converts Korean text", async () => {
		const resolved = await resolveFileeBinary({ home, env: { PATH: "/nonexistent" }, autoInstall: false });
		expect(resolved.source).toBe("cached");
		const dataDir = join(root, "filee-data");
		const options = await resolveFileeOptions(
			{ dataDir, autoInstall: false },
			{ home, env: { PATH: "/nonexistent" } },
		);
		expect(await listFileeFormats("md", options)).toContain("docx");
		const source = join(root, "회의록.md");
		writeFileSync(source, "# 회의록\n\n오늘 회의에서 예산 승인과 일정 조정을 논의했습니다.\n");
		const result = await convertWithFilee({ inputs: [source], to: "docx" }, options);
		expect(result).toMatchObject({ converted: 1, failed: 0 });
		expect(
			readFileSync(result.files[0]?.outputs[0] ?? "")
				.subarray(0, 2)
				.toString(),
		).toBe("PK");
	}, 300_000);
});

/** `ensureFileeBinary` refuses the network under NODE_ENV=test; the live run is the one place it must not. */
function nodeTestOverride(): Record<string, never> {
	process.env.NODE_ENV = "production";
	return {};
}
