import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

export const AUTORAG_HOME_ENV = "AUTORAG_HOME";

export function resolveAutoRAGHome(env: NodeJS.ProcessEnv = process.env): string {
	const configured = env[AUTORAG_HOME_ENV]?.trim();
	if (configured) return resolve(configured);
	const envHome = env.HOME ?? env.USERPROFILE;
	return join(envHome && isAbsolute(envHome) ? envHome : homedir(), ".autorag");
}
