/**
 * Orchestrates a whole search: emit the group list immediately, run every group
 * in parallel, stream one group line per group in completion order, and finish
 * with the done line. One group throwing never affects the others.
 */

import { buildGroupRunners, type FindGroupRunner } from "./groups.ts";
import type { FindGroupDescriptor, FindGroupLine, FindJudge, FindLine, FindRuntime } from "./types.ts";

export interface FindRunOptions {
	readonly query: string;
	readonly topK: number;
	readonly runtime: FindRuntime;
	readonly judge?: FindJudge;
	readonly platform?: NodeJS.Platform;
	readonly fileSearchEnabled?: boolean;
	readonly semanticEnabled?: boolean;
	/** The enabled, registered datasources, in config order. */
	readonly datasources?: readonly FindGroupDescriptor[];
	/** `--groups` narrowing: run only these keys. */
	readonly groups?: readonly string[];
	readonly emit: (line: FindLine) => void;
	readonly now?: () => number;
}

export async function runFind(options: FindRunOptions): Promise<void> {
	const now = options.now ?? Date.now;
	const runners = buildGroupRunners({
		query: options.query,
		topK: options.topK,
		runtime: options.runtime,
		...(options.judge === undefined ? {} : { judge: options.judge }),
		...(options.platform === undefined ? {} : { platform: options.platform }),
		...(options.fileSearchEnabled === undefined ? {} : { fileSearchEnabled: options.fileSearchEnabled }),
		...(options.semanticEnabled === undefined ? {} : { semanticEnabled: options.semanticEnabled }),
		...(options.datasources === undefined ? {} : { datasources: options.datasources }),
	});
	const wanted = options.groups;
	const selected = wanted === undefined ? runners : runners.filter((runner) => wanted.includes(runner.key));

	options.emit({ type: "groups", groups: selected.map((runner) => runner.descriptor) });

	const started = now();
	await Promise.all(
		selected.map(async (runner) => {
			const groupStart = now();
			options.emit(await runGroup(runner, now, groupStart));
		}),
	);
	options.emit({ type: "done", elapsedMs: now() - started });
}

async function runGroup(runner: FindGroupRunner, now: () => number, started: number): Promise<FindGroupLine> {
	try {
		const outcome = await runner.run();
		return {
			type: "group",
			key: runner.key,
			status: outcome.status,
			hits: [...outcome.hits],
			error: outcome.error,
			elapsedMs: now() - started,
		};
	} catch (error) {
		return {
			type: "group",
			key: runner.key,
			status: "error",
			hits: [],
			error: error instanceof Error ? error.message : String(error),
			elapsedMs: now() - started,
		};
	}
}
