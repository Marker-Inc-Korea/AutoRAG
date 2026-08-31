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

export type ParseDiagnosticCode = "pdf-extract-thin" | "pdf-hybrid-unavailable";

export interface ParseDiagnostic {
	readonly code: ParseDiagnosticCode;
	readonly severity: "warning";
	readonly message: string;
}

export abstract class Parser {
	abstract readonly name: string;
	abstract readonly extensions: readonly string[];

	abstract parse(input: ParseInput): Promise<ParseOutput>;
}
