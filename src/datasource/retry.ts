/**
 * Shared retry/backoff policy for datasource `index()` calls.
 *
 * Failure causes differ per datasource (HTTP status, CLI exit reason, ...), so
 * no attempt is made to classify them. {@link indexWithRetry} retries every
 * failed or thrown `index()` with jittered exponential backoff and, once the
 * retries are used up, reports that datasource as failed so the refresh moves
 * on to the next one.
 *
 * Process-tree termination of a timed-out CLI is a separate concern (#1587,
 * #1586); this module never touches the spawn code.
 */

import { setTimeout as delay } from "node:timers/promises";
import type { DatasourceDiagnostic, DatasourceIndexFail, DatasourceIndexResult, DatasourceSkill } from "./types.ts";

export interface DatasourceRetryPolicy {
	/** Re-runs after the first call; `0` disables retrying. */
	readonly maxRetries: number;
	/** Backoff before the first retry; doubles for each later retry. */
	readonly baseDelayMs: number;
	/** Upper bound on one backoff step (before jitter). */
	readonly maxDelayMs: number;
}

export interface DatasourceRetryOptions extends Partial<DatasourceRetryPolicy> {
	/** Jitter source in `[0, 1]`; injectable for deterministic tests. */
	readonly random?: () => number;
	/** Delay implementation; injectable so tests and hosts need not wait in real time. */
	readonly sleep?: (ms: number) => Promise<void>;
}

/** Two retries (three calls in total) with 1s then 2s of backoff, jittered. */
export const DEFAULT_DATASOURCE_RETRY_POLICY: DatasourceRetryPolicy = {
	maxRetries: 2,
	baseDelayMs: 1_000,
	maxDelayMs: 30_000,
};

/**
 * Backoff before retry number `retry` (1-based): `baseDelayMs * 2^(retry-1)`
 * capped at `maxDelayMs`, then jittered into the upper half of that step so
 * simultaneous refreshes do not hammer the same upstream in lockstep.
 */
export function datasourceRetryDelayMs(
	retry: number,
	policy: DatasourceRetryPolicy & { readonly random?: () => number },
): number {
	const step = Math.min(policy.maxDelayMs, policy.baseDelayMs * 2 ** (retry - 1));
	return Math.round(step * (0.5 + 0.5 * (policy.random ?? Math.random)()));
}

/**
 * Run `skill.index()`, retrying any failure (a failed result or a throw). A
 * first-time success is returned untouched. Otherwise the result carries the
 * total `attempts`; an exhausted failure also carries `attempts` in its message
 * and last diagnostic, and a late success carries an info diagnostic naming the
 * last error it recovered from. A thrown `index()` becomes a
 * `datasource-index-failed` result.
 *
 * A cancelled `signal` is not a datasource failure: it is rethrown at once
 * (also from the backoff wait) instead of being retried or reported.
 */
export async function indexWithRetry(
	skill: DatasourceSkill,
	options: DatasourceRetryOptions = {},
	signal?: AbortSignal,
): Promise<DatasourceIndexResult> {
	const policy: DatasourceRetryPolicy = {
		maxRetries: options.maxRetries ?? DEFAULT_DATASOURCE_RETRY_POLICY.maxRetries,
		baseDelayMs: options.baseDelayMs ?? DEFAULT_DATASOURCE_RETRY_POLICY.baseDelayMs,
		maxDelayMs: options.maxDelayMs ?? DEFAULT_DATASOURCE_RETRY_POLICY.maxDelayMs,
	};
	const sleep = options.sleep ?? ((ms: number) => delay(ms, undefined, signal === undefined ? undefined : { signal }));
	let lastError = "";
	for (let attempt = 1; ; attempt += 1) {
		const result = await runIndex(skill, signal);
		signal?.throwIfAborted();
		if (result.ok) {
			if (attempt === 1) return result;
			const recovered: DatasourceDiagnostic = {
				code: "datasource-index-failed",
				severity: "info",
				message: `recovered after ${attempt} attempts; last error: ${lastError}`,
				instanceId: result.instanceId,
				source: result.skill,
			};
			return { ...result, attempts: attempt, diagnostics: [...result.diagnostics, recovered] };
		}
		lastError = result.message ?? result.error ?? "datasource-index-failed";
		if (attempt > policy.maxRetries) {
			if (attempt === 1) return { ...result, attempts: attempt };
			const suffix = ` (gave up after ${attempt} attempts)`;
			const last = result.diagnostics.at(-1);
			return {
				...result,
				attempts: attempt,
				message: `${lastError}${suffix}`,
				diagnostics:
					last === undefined
						? [
								{
									code: result.code ?? "datasource-index-failed",
									severity: "error",
									message: `${lastError}${suffix}`,
									instanceId: result.instanceId,
									source: result.skill,
								},
							]
						: [...result.diagnostics.slice(0, -1), { ...last, message: `${last.message}${suffix}` }],
			};
		}
		await sleep(
			datasourceRetryDelayMs(attempt, { ...policy, ...(options.random ? { random: options.random } : {}) }),
		);
		signal?.throwIfAborted();
	}
}

async function runIndex(skill: DatasourceSkill, signal: AbortSignal | undefined): Promise<DatasourceIndexResult> {
	try {
		return await skill.index(signal === undefined ? undefined : { signal });
	} catch (error) {
		// A cancelled refresh is not a failed datasource: surface the abort instead of recording a failure.
		signal?.throwIfAborted();
		const descriptor = skill.describe();
		const message = error instanceof Error ? error.message : "Datasource indexing failed.";
		const failed: DatasourceIndexFail = {
			ok: false,
			instanceId: descriptor.instanceId ?? "default",
			skill: descriptor.name,
			indexedAt: Date.now(),
			diagnostics: [
				{
					code: "datasource-index-failed",
					severity: "error",
					message,
					source: descriptor.name,
					instanceId: descriptor.instanceId,
				},
			],
			error: "datasource-index-failed",
			code: "datasource-index-failed",
			message,
		};
		return failed;
	}
}
