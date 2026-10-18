import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll } from "vitest";

/**
 * MinSync's Korean tokenizer needs the Kiwi native library and model, and AutoRAG only selects it when they exist.
 * A test result must not depend on whether the developer's machine happens to have Kiwi installed, so every test
 * starts with stand-in files that make the probe succeed. Tests of the missing-Kiwi path override these variables.
 */
const directory = mkdtempSync(join(tmpdir(), "autorag-fake-kiwi-"));
const library = join(directory, "libkiwi");
const model = join(directory, "model");
writeFileSync(library, "");
mkdirSync(model);
process.env.KIWI_LIBRARY_PATH = library;
process.env.KIWI_MODEL_PATH = model;

afterAll(() => {
	rmSync(directory, { recursive: true, force: true });
});
