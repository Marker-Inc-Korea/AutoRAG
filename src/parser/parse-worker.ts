import { parentPort, workerData } from "node:worker_threads";
import { createDefaultParserRegistry } from "./defaults.ts";
import { ParseError } from "./errors.ts";
import type { ParseWorkerRequest, ParseWorkerResponse, WorkerParserOptions } from "./isolated.ts";

/**
 * Worker-thread entry for {@link IsolatedParseRunner}. It rebuilds the default
 * parser registry from the cloneable options in `workerData`, then parses one
 * request at a time and posts back either the output or the failure.
 */
const registry = createDefaultParserRegistry(workerData as WorkerParserOptions);

function describe(error: unknown): { readonly message: string; readonly stack?: string } {
	if (error instanceof Error)
		return { message: error.message, ...(error.stack === undefined ? {} : { stack: error.stack }) };
	return { message: typeof error === "string" ? error : "unknown parser failure" };
}

async function handle(request: ParseWorkerRequest): Promise<ParseWorkerResponse> {
	const parser = registry.list().find((candidate) => candidate.name === request.parserName);
	if (parser === undefined) {
		return { id: request.id, ok: false, kind: "error", message: `Unknown parser "${request.parserName}"` };
	}
	try {
		const output = await parser.parse({
			virtualPath: request.virtualPath,
			...(request.sourcePath === undefined ? {} : { sourcePath: request.sourcePath }),
			bytes: request.bytes,
		});
		return { id: request.id, ok: true, output };
	} catch (error) {
		if (error instanceof ParseError) {
			return {
				id: request.id,
				ok: false,
				kind: "parse-error",
				parserName: error.parserName,
				...describe(error.cause),
			};
		}
		return { id: request.id, ok: false, kind: "error", ...describe(error) };
	}
}

parentPort?.on("message", (request: ParseWorkerRequest) => {
	void handle(request).then((response) => parentPort?.postMessage(response));
});
