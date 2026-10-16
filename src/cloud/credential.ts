import { join } from "node:path";
import { getAgentDir, readStoredCredential } from "@earendil-works/pi-coding-agent";
import { PROVIDER_ID } from "./config.ts";
import { resolveCredentialKey } from "./models.ts";

/** Message for every Dazzi-backed feature used without a stored `/login` credential. */
export const DAZZI_LOGIN_REQUIRED_MESSAGE =
	"Dazzi Models needs a sign-in: run `autorag tui`, then `/login` and choose Dazzi Models.";

/**
 * The bearer token `/login` stored for the Dazzi Models plan, or `undefined`
 * when the user has not signed in.
 *
 * Login is the only credential: Dazzi's `/v1/*` endpoints take the key the
 * sign-in flow issues, so Jev and rerank read the same `auth.json` entry the
 * model provider uses. `agentDir` defaults to Pi's agent home.
 */
export function readDazziLoginToken(agentDir: string = getAgentDir()): string | undefined {
	return resolveCredentialKey(readStoredCredential(PROVIDER_ID, join(agentDir, "auth.json")));
}
