import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterAll } from "vitest";
import { writeInertMinSyncExecutable } from "../helpers/fake-minsync.ts";

/**
 * MinSync is required, so every agent a test builds needs a `minsync` binary to
 * resolve. Without this, a test result would depend on whether the developer's
 * machine happens to have a real one on PATH. The inert stand-in goes
 * first on PATH; tests that exercise a missing binary set PATH or `binaryPath`
 * themselves.
 */
const directory = mkdtempSync(join(tmpdir(), "autorag-fake-minsync-path-"));
writeInertMinSyncExecutable(directory);
const originalPath = process.env.PATH;
process.env.PATH = originalPath ? `${directory}${delimiter}${originalPath}` : directory;

afterAll(() => {
	rmSync(directory, { recursive: true, force: true });
});
