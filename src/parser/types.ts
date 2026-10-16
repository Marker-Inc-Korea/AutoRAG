export interface ParseInput {
	readonly virtualPath: string;
	readonly sourcePath?: string;
	readonly bytes: Uint8Array;
}

export interface ParseOutput {
	readonly markdown: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
	readonly diagnostics?: readonly ParseDiagnostic[];
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
	 * Stamp recorded next to a per-file skip. Refresh retries a source a parser failed on exactly
	 * once when this changes, so bump it whenever parse behavior changes (including the engine behind it).
	 */
	readonly version: string = "1";

	abstract parse(input: ParseInput): Promise<ParseOutput>;
}
