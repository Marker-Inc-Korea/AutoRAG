import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll } from "vitest";

/**
 * Defaults (Jev backend, reranker, question decomposition, chat model) follow
 * whether a Dazzi Models `/login` credential is stored under Pi's agent dir. Point
 * that dir at an empty temp directory so no test depends on the developer's own
 * `~/.pi/agent/auth.json`. Tests that exercise a signed-in state pass an explicit
 * `agentDir` or set `PI_CODING_AGENT_DIR` themselves.
 */
const directory = mkdtempSync(join(tmpdir(), "autorag-test-pi-agent-"));
process.env.PI_CODING_AGENT_DIR = directory;

afterAll(() => {
	rmSync(directory, { recursive: true, force: true });
});
