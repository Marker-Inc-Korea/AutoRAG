export interface CommandContext {
	positionals: string[];
	flags: Record<string, string | boolean | undefined>;
	json: boolean;
	debug: boolean;
	cwd: string;
	stdout: (line: string) => void;
	stderr: (line: string) => void;
	promptYesNo?: (question: string) => Promise<boolean>;
}
