import type { ParsedPages } from "./pages.ts";

export interface ParseInput {
	readonly virtualPath: string;
	readonly sourcePath?: string;
	readonly bytes: Uint8Array;
}

export interface ParseOutput {
	readonly markdown: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
	readonly diagnostics?: readonly ParseDiagnostic[];
	/** Where each page of a paged document (PDF, HWP/HWPX, PPTX) sits in `markdown`; absent for formats without pages. */
	readonly pages?: ParsedPages;
}

export type ParseDiagnosticCode = "pdf-extract-thin" | "parser-warning";

export interface ParseDiagnostic {
	readonly code: ParseDiagnosticCode;
	readonly severity: "warning";
	readonly message: string;
}

export abstract class Parser {
	abstract readonly name: string;
	abstract readonly extensions: readonly string[];
	/**
	 * Processing revision used by the durable parse stage and per-file failure skips.
	 * Bump whenever parse behavior changes, including an engine dependency upgrade.
	 */
	readonly version: string = "1";

	abstract parse(input: ParseInput): Promise<ParseOutput>;
}
