/**
 * Polling helpers for datasource skills.
 *
 * v1 supports a simple interval-based `poll` mode plus a cron-expression
 * validator. Cron is **descriptor-only**: {@link parseCronExpr} validates
 * syntax so a skill descriptor can declare a schedule, but this module never
 * schedules or fires jobs — scheduling is a caller/server concern.
 */

import type { PollingMetadata } from "./types.ts";

/**
 * Whether a datasource skill/instance is due for a poll/index at `now`.
 *
 * Rules (v1, poll mode only):
 *  - `mode !== "poll"` ⇒ never due (no automatic polling).
 *  - `intervalMs` missing or non-positive ⇒ never due (misconfigured).
 *  - never indexed (`lastIndexedAt` unknown) ⇒ due immediately.
 *  - otherwise due iff `now - lastIndexedAt >= intervalMs`.
 *
 * `lastIndexedAt` is taken from the explicit argument when provided,
 * otherwise from {@link PollingMetadata.lastIndexedAt}; this lets a caller
 * pass an authoritative value that may differ from stale metadata.
 */
export function isDue(metadata: PollingMetadata, now: number, lastIndexedAt?: number): boolean {
	if (metadata.mode !== "poll") return false;
	const intervalMs = metadata.intervalMs;
	if (intervalMs === undefined || intervalMs <= 0) return false;
	const last = lastIndexedAt ?? metadata.lastIndexedAt;
	if (last === undefined) return true;
	if (!Number.isFinite(now) || !Number.isFinite(last)) return false;
	return now - last >= intervalMs;
}

/** Successful cron parse result. */
export interface CronParseOk {
	readonly ok: true;
	/** The normalized (trimmed) expression. */
	readonly expr: string;
}

/** Failed cron parse result with a human-readable error. */
export interface CronParseFail {
	readonly ok: false;
	readonly error: string;
}

/** Result of {@link parseCronExpr}. */
export type CronParseResult = CronParseOk | CronParseFail;

/**
 * Validate a 5-field cron expression.
 *
 * Fields: `minute hour day-of-month month day-of-week`.
 *
 * Supported syntax per field: star, star-slash-n, n, a-b, a-b-slash-n,
 * and comma lists of the above (e.g. 1,15,30). Extended forms (L, W, ?,
 * NOT supported and will fail validation.
 *
 * Ranges: minute 0–59, hour 0–23, day-of-month 1–31, month 1–12, day-of-week
 * 0–7 (where both 0 and 7 denote Sunday).
 *
 * This function only validates syntax. It does not schedule anything.
 */
export function parseCronExpr(expr: string): CronParseResult {
	if (typeof expr !== "string") {
		return fail("cron expression must be a string");
	}
	const trimmed = expr.trim();
	if (trimmed.length === 0) {
		return fail("cron expression is empty");
	}
	const fields = trimmed.split(/\s+/u);
	if (fields.length !== 5) {
		return fail(`expected 5 fields, got ${fields.length}`);
	}
	const [minute, hour, dom, month, dow] = fields;
	const errors: string[] = [];
	validateField(minute, 0, 59, "minute", errors);
	validateField(hour, 0, 23, "hour", errors);
	validateField(dom, 1, 31, "day-of-month", errors);
	validateField(month, 1, 12, "month", errors);
	validateField(dow, 0, 7, "day-of-week", errors);
	if (errors.length > 0) {
		return fail(errors.join("; "));
	}
	return { ok: true, expr: trimmed };
}

function fail(error: string): CronParseResult {
	return { ok: false, error };
}

function validateField(field: string, min: number, max: number, label: string, errors: string[]): void {
	const parts = field.split(",");
	if (parts.length === 0) {
		errors.push(`${label}: empty`);
		return;
	}
	for (const part of parts) {
		const err = validatePart(part, min, max, label);
		if (err !== null) errors.push(err);
	}
}

function validatePart(part: string, min: number, max: number, label: string): string | null {
	if (part.length === 0) return `${label}: empty term`;

	// Step form: base/step  (e.g. */15, 1-5/2, 10/2).
	const slashIndex = part.indexOf("/");
	let base = part;
	let stepText: string | undefined;
	if (slashIndex >= 0) {
		base = part.slice(0, slashIndex);
		stepText = part.slice(slashIndex + 1);
		const step = parseNumber(stepText);
		if (step === null || step <= 0) {
			return `${label}: invalid step "${stepText}"`;
		}
	}

	if (base === "*") return null;

	if (base.includes("-")) {
		const [loText, hiText] = base.split("-");
		if (hiText === undefined || loText === undefined) {
			return `${label}: malformed range "${base}"`;
		}
		const lo = parseNumber(loText);
		const hi = parseNumber(hiText);
		if (lo === null || hi === null) {
			return `${label}: non-numeric range "${base}"`;
		}
		if (lo < min || lo > max || hi < min || hi > max) {
			return `${label}: range "${base}" out of bounds (${min}-${max})`;
		}
		if (lo > hi) {
			return `${label}: range "${base}" has low > high`;
		}
		return null;
	}

	const v = parseNumber(base);
	if (v === null) {
		return `${label}: invalid value "${base}"`;
	}
	if (v < min || v > max) {
		return `${label}: value ${v} out of bounds (${min}-${max})`;
	}
	return null;
}

function parseNumber(text: string): number | null {
	if (!/^\d+$/u.test(text)) return null;
	return Number.parseInt(text, 10);
}
