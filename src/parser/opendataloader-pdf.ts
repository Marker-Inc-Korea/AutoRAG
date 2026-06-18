import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, parse } from "node:path";
import { convert } from "@opendataloader/pdf";
import { ParseError } from "./errors.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";

export class OpendataloaderPdfParser extends Parser {
	readonly name = "opendataloader-pdf";
	readonly extensions = [".pdf"] as const;

	async parse(input: ParseInput): Promise<ParseOutput> {
		const tempRoot = await mkdtemp(join(tmpdir(), "autorag-pdf-"));
		try {
			const inputPath = input.sourcePath ?? join(tempRoot, basename(input.virtualPath));
			if (!input.sourcePath) {
				await writeFile(inputPath, input.bytes);
			}

			await convert(inputPath, {
				outputDir: tempRoot,
				format: "markdown",
				quiet: true,
				imageOutput: "off",
			});

			return {
				markdown: await readFile(markdownOutputPath(tempRoot, inputPath), "utf8"),
				metadata: { parser: this.name },
			};
		} catch (cause) {
			throw new ParseError(this.name, input.virtualPath, cause);
		} finally {
			await rm(tempRoot, { recursive: true, force: true });
		}
	}
}

function markdownOutputPath(outputDir: string, inputPath: string): string {
	return join(outputDir, `${parse(inputPath).name}.md`);
}
