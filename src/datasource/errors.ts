/**
 * Datasource retrieval errors.
 *
 * A CLI-backed datasource that cannot answer must say why. Its retrieval method
 * throws this error instead of returning an empty result set, so the retrieval
 * pipeline reports the surface as unsearched and hands the operator the CLI's
 * own words: failure kind, exit status, and stderr verbatim.
 */

/** The failure payload every datasource CLI client returns (katok, discrawl, qmd, …). */
export interface DatasourceCliFailure {
	readonly reason: string;
	readonly stderr?: string;
	readonly stdout?: string;
	readonly code?: number | null;
}

/** Error thrown by a datasource retrieval method when its CLI failed. */
export class DatasourceCliError extends Error {
	readonly datasourceId: string;
	readonly command: string;
	readonly reason: string;
	readonly stderr: string;
	readonly code: number | null;

	constructor(datasourceId: string, command: string, failure: DatasourceCliFailure) {
		const stderr = (failure.stderr ?? "").trim();
		const detail = stderr.length > 0 ? stderr : (failure.stdout ?? "").trim();
		const exit = failure.code === undefined || failure.code === null ? "no exit code" : `exit code ${failure.code}`;
		super(`${datasourceId} ${command} failed (${failure.reason}, ${exit})${detail.length > 0 ? `: ${detail}` : ""}`);
		this.name = "DatasourceCliError";
		this.datasourceId = datasourceId;
		this.command = command;
		this.reason = failure.reason;
		this.stderr = stderr;
		this.code = failure.code ?? null;
	}
}

/** Build the error a datasource retrieval method throws for a failed CLI call. */
export function datasourceCliError(
	datasourceId: string,
	command: string,
	failure: DatasourceCliFailure,
): DatasourceCliError {
	return new DatasourceCliError(datasourceId, command, failure);
}
