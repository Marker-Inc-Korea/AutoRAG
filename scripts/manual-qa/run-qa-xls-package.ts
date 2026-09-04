import { readFile } from "node:fs/promises";
import { basename } from "node:path";
import { XlsxParser } from "../../dist/index.js";

const inputPath = process.argv[2];
if (inputPath === undefined) {
	throw new Error("usage: bun scripts/manual-qa/run-qa-xls-package.ts <path-to-xls>");
}

const parsed = await new XlsxParser().parse({
	virtualPath: basename(inputPath),
	sourcePath: inputPath,
	bytes: await readFile(inputPath),
});

if (parsed.metadata?.format !== "xls") {
	throw new Error("expected legacy XLS metadata");
}
process.stdout.write(`${parsed.markdown}\n`);
