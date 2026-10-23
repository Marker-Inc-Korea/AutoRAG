import { createRequire } from "node:module";
import { dirname, join } from "node:path";

/** Tool the `pi-ask-user` extension registers; pi's allow-list must name it to keep it active. */
export const ASK_USER_TOOL_NAME = "ask_user";

/**
 * Entry file of the community `pi-ask-user` extension (the most-installed pi package for this: Claude
 * Code's AskUserQuestion, with a searchable single/multi-select sheet, freeform answers, and 2-4 question
 * batches). It ships TypeScript source and no `main`, so pi's own extension loader runs it from this path.
 */
export function resolveAskUserExtensionPath(): string {
	const manifest = createRequire(import.meta.url).resolve("pi-ask-user/package.json");
	return join(dirname(manifest), "index.ts");
}
