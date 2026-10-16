export interface CommandContext {
	positionals: string[];
	flags: Record<string, string | boolean | undefined>;
	/** All values of repeatable flags, in command-line order. */
	repeatedFlags?: Readonly<Record<string, readonly string[]>>;
	json: boolean;
	debug: boolean;
	cwd: string;
	stdout: (line: string) => void;
	stderr: (line: string) => void;
	promptYesNo?: (question: string) => Promise<boolean>;
}
