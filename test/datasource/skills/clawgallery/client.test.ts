import { chmodSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { ClawGalleryClient } from "../../../../src/datasource/skills/clawgallery/client.ts";

function stubBinary(script: string): string {
	const dir = mkdtempSync(join(tmpdir(), "clawgallery-stub-"));
	const path = join(dir, "clawgallery");
	writeFileSync(path, `#!/bin/sh\n${script}\n`, "utf8");
	chmodSync(path, 0o755);
	return path;
}

describe("ClawGalleryClient", () => {
	it("runs incremental bootstrap and visual sync with trusted backend", async () => {
		const binaryPath = stubBinary(
			'case "$1 $2" in "bootstrap ") echo "ingested 1 new image(s)" ;; "vdr sync") echo "indexed 1 image(s)" ;; esac',
		);
		const client = new ClawGalleryClient({ binaryPath, vdrBackend: "vsplade" });
		expect(await client.bootstrap()).toMatchObject({ ok: true, data: { indexed: 1, skipped: 0 } });
		expect(await client.syncVisual()).toMatchObject({ ok: true, data: { processed: 1 } });
	});

	it("maps JSON search hits and preserves the selected mode", async () => {
		const binaryPath = stubBinary(
			'printf \'{"id":"img-1","path":"/tmp/login.png","caption":"Login error","score":0.9}\\n\'',
		);
		const result = await new ClawGalleryClient({ binaryPath }).search("embedding", "login error", { topK: 3 });
		expect(result).toMatchObject({
			ok: true,
			hits: [{ imageId: "img-1", path: "/tmp/login.png", caption: "Login error" }],
		});
	});

	it("isolates child environment to ClawGallery settings", async () => {
		const binaryPath = stubBinary(
			'printf \'{"env":"%s,%s,%s"}\' "$OPENAI_API_KEY" "$CLAWGALLERY_PYTHON" "$CLAWGALLERY_CONFIG_DIR"',
		);
		const result = await new ClawGalleryClient({
			binaryPath,
			configDir: "/tmp/gallery-config",
			env: { CLAWGALLERY_PYTHON: "/tmp/python", OPENAI_API_KEY: "secret" },
		}).bootstrap();
		expect(result).toMatchObject({ ok: false, reason: "invalid-shape" });
		expect(result.stdout).toContain("/tmp/python,/tmp/gallery-config");
		expect(result.stdout).not.toContain("secret");
	});

	it("reports a missing binary without throwing", async () => {
		await expect(
			new ClawGalleryClient({ binaryPath: "/nonexistent/clawgallery" }).bootstrap(),
		).resolves.toMatchObject({
			ok: false,
			reason: "binary-missing",
		});
	});
});
