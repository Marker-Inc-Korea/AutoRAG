export function configuredModelEnvNames(): readonly string[];
export function parseRunnerArgs(input: string): readonly string[];
export function containerEnvironment(): {
	readonly fixed: readonly (readonly [name: string, value: string])[];
	readonly modelEnv: readonly string[];
};