import { spawn } from "node:child_process";

/** Command a user runs to provide the dupey CLI this app requires. */
export const DUPEY_INSTALL_COMMAND = "cargo install dupey --locked";

export interface DupeyStatus {
	readonly available: boolean;
	readonly version: string | null;
	readonly error: string | null;
}

export interface DupeyProbe {
	status(): Promise<DupeyStatus>;
}

export type DupeyVersionRunner = () => Promise<string>;

function runDupeyVersion(): Promise<string> {
	return new Promise((resolve, reject) => {
		const child = spawn("dupey", ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (chunk: Buffer) => {
			stdout += String(chunk);
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += String(chunk);
		});
		child.on("error", (error: Error) => reject(error));
		child.on("close", (code: number | null) => {
			if (code === 0) {
				resolve(stdout.trim());
				return;
			}
			reject(new Error(stderr.trim() || `dupey --version exited with code ${code ?? "null"}`));
		});
	});
}

export interface DupeyProbeOptions {
	readonly run?: DupeyVersionRunner;
	/** How long a probe result stays fresh, so a just-installed dupey is picked up. */
	readonly ttlMs?: number;
	readonly now?: () => number;
}

export function createDupeyProbe(options: DupeyProbeOptions = {}): DupeyProbe {
	const run = options.run ?? runDupeyVersion;
	const ttlMs = options.ttlMs ?? 30_000;
	const now = options.now ?? (() => Date.now());
	let cached: { readonly at: number; readonly status: DupeyStatus } | null = null;

	return {
		async status(): Promise<DupeyStatus> {
			if (cached !== null && now() - cached.at < ttlMs) return cached.status;
			let status: DupeyStatus;
			try {
				const output = await run();
				status = { available: true, version: output.length > 0 ? output : null, error: null };
			} catch (error) {
				status = {
					available: false,
					version: null,
					error: error instanceof Error ? error.message : String(error),
				};
			}
			cached = { at: now(), status };
			return status;
		},
	};
}
