import type { ExtensionAPI, ExtensionFactory } from "@earendil-works/pi-coding-agent";

export const FAST_GREP_ALLOWED_TOOLS = new Set(["grep", "read", "find", "ls"]);

export const DEFAULT_FAST_GREP_TOOL_CAP = 5;

export interface FastPhaseParallelPrefetchOptions {
	/**
	 * Maximum fast-phase exploration tool calls (`grep`/`read`/`find`/`ls`) allowed before the
	 * harness waits for the reranked baseline, injects it, and forces an immediate fast answer.
	 * Defaults to {@link DEFAULT_FAST_GREP_TOOL_CAP}.
	 */
	readonly toolCap?: number;
}

export interface FastPhaseParallelState {
	/** Resolves once the reranked MinSync + datasource + web baseline is ready. */
	readonly baselinePromise: Promise<string>;
	/** Set as soon as {@link baselinePromise} fulfills. */
	baselineReady: string | undefined;
	/** True once {@link baselinePromise} has rejected or failed cleanly. */
	baselineSettled: boolean;
	/** True once the baseline has been injected into this fast phase (via tool_result or follow-up). */
	baselineInjected: boolean;
	/** Number of fast-phase exploration tool calls executed so far. */
	explorationCalls: number;
	/** Maximum exploration tool calls before forcing baseline wait + stop. */
	readonly toolCap: number;
	/** Diagnostic reason when the fast phase stopped exploration. */
	stopReason?: "baseline-injected" | "tool-cap" | "fast-answered";
	/** Optional run-abort signal so waiting on `baselinePromise` never hangs past user abort. */
	readonly abortSignal?: AbortSignal;
}

function awaitBaselineOrAbort(baselinePromise: Promise<string>, signal?: AbortSignal): Promise<string> {
	if (signal === undefined) return baselinePromise.catch(() => "");
	if (signal.aborted) return Promise.resolve("");
	return new Promise<string>((resolve) => {
		const onAbort = (): void => resolve("");
		signal.addEventListener("abort", onAbort, { once: true });
		void baselinePromise.then(
			(value) => {
				signal.removeEventListener("abort", onAbort);
				resolve(value);
			},
			() => {
				signal.removeEventListener("abort", onAbort);
				resolve("");
			},
		);
	});
}

export function buildInjectedBaselineBlock(baseline: string, reason: "arrived" | "tool-cap"): string {
	const header =
		reason === "tool-cap"
			? `[Baseline retrieval evidence arrived — exploration limit reached]`
			: `[Baseline retrieval evidence arrived]`;
	return (
		`\n\n---\n${header}\n` +
		`The reranked MinSync + datasource baseline for this query is now ready:\n\n` +
		`${baseline}\n\n` +
		`IMPORTANT: Stop calling further search/read/grep tools now. Write your immediate fast answer as plain text with inline [eN] citations from the confirmed evidence above (if unconfirmed, reply with one short line stating it is not confirmed yet and do not pad with unrelated hits or searched keywords). Never mention internal search mechanics (grep, baseline arrival, stopped tools, or verification passes) in your answer.`
	);
}

/**
 * Extension that coordinates the parallel grep-first + reranked-MinSync fast phase.
 *
 * Lifecycle per fast-phase turn:
 * - While `getState()` returns `undefined` (verification phase, action turns, or legacy sequential
 *   fast phase), every hook is a complete no-op.
 * - While `getState()` is active:
 *   1. Before a tool call executes (`tool_call`):
 *      - Only lightweight mirror exploration (`grep`, `read`, `find`, `ls`) and `calculator` /
 *        `web_search` / `web_fetch` / `ask_user` are permitted.
 *      - If the reranked baseline has ALREADY been injected, any further exploration tool call is
 *        blocked with a concise directive to stop calling tools and emit the fast answer now.
 *      - If `explorationCalls >= toolCap` and the baseline has not yet been injected, the hook
 *        awaits `baselinePromise`, injects the reranked baseline directly into the block reason,
 *        marks `baselineInjected = true`, and instructs the model to emit the fast answer now.
 *   2. After an exploration tool finishes (`tool_result`):
 *      - Increments `explorationCalls`.
 *      - If `baselineReady !== undefined` and `!baselineInjected`, appends the reranked baseline
 *        block to this tool result, marks `baselineInjected = true`, and tells the model to stop
 *        calling tools and produce the fast answer immediately.
 *      - Else if `explorationCalls >= toolCap` and `!baselineInjected`, awaits `baselinePromise`,
 *        appends the reranked baseline block to this tool result, and marks `baselineInjected = true`.
 */
export function createFastPhaseParallelExtension(getState: () => FastPhaseParallelState | undefined): ExtensionFactory {
	return (pi: ExtensionAPI) => {
		pi.on("tool_call", async (event) => {
			const state = getState();
			if (state === undefined) return;
			const toolName = event.toolName;
			// Non-exploration tools that are always allowed in fast phase (calculator, web, ask_user).
			if (
				toolName === "calculator" ||
				toolName === "web_search" ||
				toolName === "web_fetch" ||
				toolName === "ask_user"
			) {
				return;
			}
			// Heavy retrieval / execution tools are not allowed in the fast phase (verification handles them).
			if (!FAST_GREP_ALLOWED_TOOLS.has(toolName)) {
				return {
					block: true,
					reason:
						`Tool "${toolName}" is disabled during the fast pass. Use only grep/read/find/ls or answer directly; ` +
						`the reranked MinSync + datasource baseline is running in parallel and a verification pass will run if needed.`,
				};
			}
			// Once the reranked baseline has already been injected, stop further tool calls and force the fast answer.
			if (state.baselineInjected) {
				return {
					block: true,
					reason:
						`Reranked MinSync + datasource baseline has already been delivered in this fast pass. ` +
						`Do NOT call more tools now — emit your best immediate fast answer as plain text with [eN] citations.`,
				};
			}
			// If the reranked baseline arrived between turns before this tool call, inject it now and halt exploration.
			if (state.baselineReady !== undefined) {
				state.baselineInjected = true;
				state.stopReason = "baseline-injected";
				return {
					block: true,
					reason: buildInjectedBaselineBlock(state.baselineReady, "arrived"),
				};
			}
			// Cap reached before the baseline arrived: wait for the reranked baseline, deliver it, and stop exploration.
			if (state.explorationCalls >= state.toolCap) {
				const baseline = await awaitBaselineOrAbort(state.baselinePromise, state.abortSignal);
				state.baselineReady = baseline;
				state.baselineSettled = true;
				state.baselineInjected = true;
				state.stopReason = "tool-cap";
				return {
					block: true,
					reason: buildInjectedBaselineBlock(baseline, "tool-cap"),
				};
			}
		});

		pi.on("tool_result", async (event) => {
			const state = getState();
			if (state === undefined) return;
			if (!FAST_GREP_ALLOWED_TOOLS.has(event.toolName)) return;
			state.explorationCalls += 1;
			if (state.baselineInjected) return;

			let injectionReason: "arrived" | "tool-cap" | undefined;
			let baseline: string | undefined;
			if (state.baselineReady !== undefined) {
				baseline = state.baselineReady;
				injectionReason = "arrived";
			} else if (state.explorationCalls >= state.toolCap) {
				baseline = await awaitBaselineOrAbort(state.baselinePromise, state.abortSignal);
				state.baselineReady = baseline;
				state.baselineSettled = true;
				injectionReason = "tool-cap";
			}
			if (baseline === undefined || injectionReason === undefined) return;

			state.baselineInjected = true;
			state.stopReason = injectionReason === "tool-cap" ? "tool-cap" : "baseline-injected";
			const block = buildInjectedBaselineBlock(baseline, injectionReason);
			return {
				content: [...event.content, { type: "text" as const, text: block }],
			};
		});
	};
}
