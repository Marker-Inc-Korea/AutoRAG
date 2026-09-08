export type DatasourceStatus = "PASS" | "SKIP" | "FAIL";
export type DatasourceLane = {
	readonly name: string;
	readonly status: DatasourceStatus;
	readonly reason: string;
	readonly evidence?: Readonly<Record<string, unknown>>;
};
export function buildDatasourceMatrix(): readonly Readonly<Record<string, string>>[];
export function parseDatasourceSelection(value?: string): readonly string[];
export function validateNativeIdentity(source: unknown, laneName?: string): boolean;
export function sanitizeDiagnostic(value: unknown): string;
export function runDatasourceMatrix(options?: Readonly<{
	readonly root?: string;
	readonly selection?: readonly string[];
	readonly which?: (binary: string) => boolean;
	readonly configured?: (name: string) => boolean;
	readonly run?: (command: string, args: readonly string[], cwd: string) => Promise<{ readonly ok: boolean; readonly stdout: string; readonly stderr: string; readonly code: number }>;
}>): Promise<Readonly<{
	readonly selection: readonly string[];
	readonly lanes: readonly DatasourceLane[];
	readonly summary: Readonly<{ readonly pass: number; readonly skip: number; readonly fail: number }>;
	readonly exitCode: number;
}>>;
