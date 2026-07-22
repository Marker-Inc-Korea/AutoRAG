export interface ParseInput {
	readonly virtualPath: string;
	readonly sourcePath?: string;
	readonly bytes: Uint8Array;
}

export interface ParseOutput {
	readonly markdown: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export abstract class Parser {
	abstract readonly name: string;
	abstract readonly extensions: readonly string[];

	abstract parse(input: ParseInput): Promise<ParseOutput>;
}
