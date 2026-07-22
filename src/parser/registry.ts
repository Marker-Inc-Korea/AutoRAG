import { extname } from "node:path";
import type { Parser } from "./types.ts";

export class ParserRegistry {
	private readonly parsersByExtension = new Map<string, Parser>();

	constructor(parsers: readonly Parser[] = []) {
		for (const parser of parsers) {
			this.register(parser);
		}
	}

	register(parser: Parser): void {
		for (const extension of parser.extensions) {
			const normalized = normalizeExtension(extension);
			if (this.parsersByExtension.has(normalized)) {
				throw new Error(`Parser extension "${normalized}" is already registered`);
			}
			this.parsersByExtension.set(normalized, parser);
		}
	}

	getForVirtualPath(virtualPath: string): Parser | undefined {
		return this.parsersByExtension.get(normalizeExtension(extname(virtualPath)));
	}

	list(): Parser[] {
		return Array.from(new Set(this.parsersByExtension.values()));
	}
}

function normalizeExtension(extension: string): string {
	const lower = extension.toLowerCase();
	return lower.startsWith(".") ? lower : `.${lower}`;
}
