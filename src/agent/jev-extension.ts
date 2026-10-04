import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { createJevEvaluator } from "../jev/evaluator.ts";
import { createJevTool } from "../jev/tool.ts";
import type { JevToolOptions } from "../jev/types.ts";
import { toToolDefinition } from "./pi-session.ts";

/**
 * Builds the pi extension that registers the optional `jev` decision tool.
 *
 * Jev is TypeSafe's judgment model: typed questions about a state in,
 * calibrated probabilities out — no generated text. Registering it as a pi
 * extension (rather than an AutoRAG custom tool) keeps the tool on pi's
 * extension surface: pi owns activation, rendering, and the `tools`
 * allow-list, and AutoRAG only declares the tool name for its prompt.
 *
 * The evaluator is lazy — no credentials are resolved when the extension
 * loads, only on the first `jev` call.
 */
export function createJevExtension(options: JevToolOptions = {}): ExtensionFactory {
	return (pi) => {
		pi.registerTool(toToolDefinition(createJevTool(createJevEvaluator(options))));
	};
}
