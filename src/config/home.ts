import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

export const AUTORAG_HOME_ENV = "AUTORAG_HOME";

/**
 * Operator home directory. `HOME` wins over `USERPROFILE` on Windows, and a
 * missing, empty, or relative value falls back to `os.homedir()` so callers
 * never build absolute-looking state from a relative base.
 */
export function resolveUserHome(env: NodeJS.ProcessEnv = process.env): string {
	const envHome = env.HOME ?? env.USERPROFILE;
	return envHome && isAbsolute(envHome) ? envHome : homedir();
}

export function resolveAutoRAGHome(env: NodeJS.ProcessEnv = process.env): string {
	const configured = env[AUTORAG_HOME_ENV]?.trim();
	if (configured) return resolve(configured);
	return join(resolveUserHome(env), ".autorag");
}
