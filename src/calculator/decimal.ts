import { CalculatorError } from "./errors.ts";

/** Significant digits every intermediate result keeps. */
export const WORK_DIGITS = 60;
/** Significant digits a final result keeps. */
export const RESULT_DIGITS = 40;
/** Decimal exponents beyond this (either sign) are refused: they only come from runaway powers. */
const MAX_ADJUSTED_EXPONENT = 100_000;
/** An addend further than this many digits below the other cannot change a WORK_DIGITS result. */
const NEGLIGIBLE_GAP = WORK_DIGITS + 70;
/** Widest span `sumExact` aligns, so one huge exponent cannot allocate a huge integer. */
const MAX_SUM_SPAN = 2_000;
export const MAX_INTEGER_EXPONENT = 10_000;

/** `n × 10^e`. Normalized: no trailing zeros in `n`, and zero is `{0n, 0}`. */
export interface Dec {
	readonly n: bigint;
	readonly e: number;
}

/** Whether any step rounded or approximated, so the answer is not the exact mathematical value. */
export interface Precision {
	inexact: boolean;
}

export const ZERO: Dec = { n: 0n, e: 0 };
export const ONE: Dec = { n: 1n, e: 0 };

export type RoundingMode = "half-away" | "floor" | "ceil" | "trunc";

function abs(n: bigint): bigint {
	return n < 0n ? -n : n;
}

export function digitCount(n: bigint): number {
	return abs(n).toString().length;
}

export function normalize(n: bigint, e: number): Dec {
	if (n === 0n) return ZERO;
	const digits = abs(n).toString();
	const trailing = digits.length - digits.replace(/0+$/u, "").length;
	const result = trailing === 0 ? { n, e } : { n: n / 10n ** BigInt(trailing), e: e + trailing };
	if (Math.abs(result.e + digitCount(result.n) - 1) > MAX_ADJUSTED_EXPONENT) {
		throw new CalculatorError("The result is out of range (more than 100000 digits before or after the point).");
	}
	return result;
}

export function parseDecimal(text: string): Dec {
	const match = /^(\d*)\.?(\d*)(?:[eE]([+-]?\d+))?$/u.exec(text);
	if (match === null || (match[1] === "" && match[2] === "")) {
		throw new CalculatorError(`"${text}" is not a number.`);
	}
	const exponent = match[3] === undefined ? 0 : Number(match[3]);
	if (!Number.isSafeInteger(exponent) || Math.abs(exponent) > MAX_ADJUSTED_EXPONENT * 2) {
		throw new CalculatorError(`The exponent in "${text}" is out of range.`);
	}
	return normalize(BigInt(`${match[1]}${match[2]}`), exponent - (match[2] ?? "").length);
}

export function fromInteger(value: number): Dec {
	return normalize(BigInt(value), 0);
}

/** A finite double rounded to 15 significant digits: transcendental results are only that accurate. */
export function fromDouble(value: number, precision: Precision, name: string): Dec {
	if (!Number.isFinite(value)) throw new CalculatorError(`${name} has no finite result for that input.`);
	precision.inexact = true;
	if (value === 0) return ZERO;
	const rounded = parseDecimal(Math.abs(value).toPrecision(15));
	return value < 0 ? negate(rounded) : rounded;
}

export function toDouble(value: Dec): number {
	return Number(`${value.n}e${value.e}`);
}

export function sign(value: Dec): -1 | 0 | 1 {
	return value.n === 0n ? 0 : value.n < 0n ? -1 : 1;
}

export function negate(value: Dec): Dec {
	return value.n === 0n ? value : { n: -value.n, e: value.e };
}

export function isInteger(value: Dec): boolean {
	return value.e >= 0;
}

/** Exponent of the leading digit: 1234.5 → 3, 0.05 → -2. */
function adjustedExponent(value: Dec): number {
	return digitCount(value.n) - 1 + value.e;
}

/** `|a| / divisor` truncated, with how the dropped part compares to half the divisor. */
function divideMagnitude(a: bigint, divisor: bigint): { quotient: bigint; remainder: bigint; half: -1 | 0 | 1 } {
	const quotient = a / divisor;
	const remainder = a % divisor;
	const doubled = remainder * 2n;
	return { quotient, remainder, half: doubled < divisor ? -1 : doubled === divisor ? 0 : 1 };
}

/** Drops every digit below `10^target`, rounding as `mode` says. */
export function roundAtExponent(value: Dec, target: number, mode: RoundingMode, precision: Precision): Dec {
	if (value.e >= target) return value;
	const drop = target - value.e;
	const negative = value.n < 0n;
	const magnitude = abs(value.n);
	let quotient: bigint;
	let lost: boolean;
	let half: -1 | 0 | 1;
	if (drop > digitCount(value.n) + 1) {
		quotient = 0n;
		lost = true;
		half = -1;
	} else {
		const parts = divideMagnitude(magnitude, 10n ** BigInt(drop));
		quotient = parts.quotient;
		lost = parts.remainder !== 0n;
		half = parts.half;
	}
	if (lost) {
		precision.inexact = true;
		const bumps =
			mode === "half-away" ? half >= 0 : mode === "floor" ? negative : mode === "ceil" ? !negative : false;
		if (bumps) quotient += 1n;
	}
	return normalize(negative ? -quotient : quotient, target);
}

/** Rounds to `digits` significant digits, half away from zero. */
export function roundSignificant(value: Dec, digits: number, precision: Precision): Dec {
	const count = digitCount(value.n);
	if (count <= digits) return value;
	return roundAtExponent(value, value.e + count - digits, "half-away", precision);
}

export function compare(a: Dec, b: Dec): -1 | 0 | 1 {
	const signA = sign(a);
	const signB = sign(b);
	if (signA !== signB) return signA < signB ? -1 : 1;
	if (signA === 0) return 0;
	const adjustedA = adjustedExponent(a);
	const adjustedB = adjustedExponent(b);
	if (adjustedA !== adjustedB) return adjustedA < adjustedB === signA > 0 ? -1 : 1;
	const low = Math.min(a.e, b.e);
	const left = a.n * 10n ** BigInt(a.e - low);
	const right = b.n * 10n ** BigInt(b.e - low);
	return left < right ? -1 : left > right ? 1 : 0;
}

export function add(a: Dec, b: Dec, precision: Precision): Dec {
	if (a.n === 0n) return b;
	if (b.n === 0n) return a;
	const gap = adjustedExponent(a) - adjustedExponent(b);
	if (gap > NEGLIGIBLE_GAP) {
		precision.inexact = true;
		return a;
	}
	if (gap < -NEGLIGIBLE_GAP) {
		precision.inexact = true;
		return b;
	}
	const low = Math.min(a.e, b.e);
	const sum = a.n * 10n ** BigInt(a.e - low) + b.n * 10n ** BigInt(b.e - low);
	return roundSignificant(normalize(sum, low), WORK_DIGITS, precision);
}

export function subtract(a: Dec, b: Dec, precision: Precision): Dec {
	return add(a, negate(b), precision);
}

export function multiply(a: Dec, b: Dec, precision: Precision): Dec {
	return roundSignificant(normalize(a.n * b.n, a.e + b.e), WORK_DIGITS, precision);
}

export function divide(a: Dec, b: Dec, precision: Precision): Dec {
	if (b.n === 0n) throw new CalculatorError("Division by zero.");
	if (a.n === 0n) return ZERO;
	const shift = Math.max(0, WORK_DIGITS + 3 + digitCount(b.n) - digitCount(a.n));
	const numerator = a.n * 10n ** BigInt(shift);
	const quotient = numerator / b.n;
	if (numerator % b.n !== 0n) {
		precision.inexact = true;
		// A sticky digit keeps a truncated quotient from rounding as if it were exactly on a half boundary.
		return roundSignificant(
			normalize(quotient * 10n + (quotient < 0n ? -1n : 1n), a.e - b.e - shift - 1),
			WORK_DIGITS,
			precision,
		);
	}
	return roundSignificant(normalize(quotient, a.e - b.e - shift), WORK_DIGITS, precision);
}

/** `a mod b` with the sign of `a`, like JavaScript's `%` but exact. */
export function remainder(a: Dec, b: Dec, precision: Precision): Dec {
	if (b.n === 0n) throw new CalculatorError("Division by zero.");
	if (a.n === 0n) return ZERO;
	if (adjustedExponent(a) - adjustedExponent(b) > MAX_SUM_SPAN) {
		throw new CalculatorError("The remainder operands differ too much in size.");
	}
	const low = Math.min(a.e, b.e);
	const result = (a.n * 10n ** BigInt(a.e - low)) % (b.n * 10n ** BigInt(b.e - low));
	return roundSignificant(normalize(result, low), WORK_DIGITS, precision);
}

/** Exact total of every term; only the final result is rounded. */
export function sumExact(terms: readonly Dec[], precision: Precision): Dec {
	const nonZero = terms.filter((term) => term.n !== 0n);
	if (nonZero.length === 0) return ZERO;
	const low = Math.min(...nonZero.map((term) => term.e));
	const high = Math.max(...nonZero.map(adjustedExponent));
	if (high - low > MAX_SUM_SPAN) throw new CalculatorError("The values span too wide a range to add exactly.");
	let total = 0n;
	for (const term of nonZero) total += term.n * 10n ** BigInt(term.e - low);
	return roundSignificant(normalize(total, low), WORK_DIGITS, precision);
}

function integerSquareRoot(value: bigint): bigint {
	if (value < 2n) return value;
	let x = 10n ** BigInt(Math.ceil(digitCount(value) / 2));
	for (;;) {
		const next = (x + value / x) >> 1n;
		if (next >= x) return x;
		x = next;
	}
}

function integerCubeRoot(value: bigint): bigint {
	if (value < 2n) return value;
	let x = 10n ** BigInt(Math.ceil(digitCount(value) / 3));
	for (;;) {
		const next = (2n * x + value / (x * x)) / 3n;
		if (next >= x) return x;
		x = next;
	}
}

/** Square root to WORK_DIGITS digits; exact when the root is a terminating decimal. */
export function squareRoot(value: Dec, precision: Precision): Dec {
	if (value.n < 0n) throw new CalculatorError("sqrt of a negative number is not a real number.");
	if (value.n === 0n) return ZERO;
	let { n, e } = value;
	if (e % 2 !== 0) {
		n *= 10n;
		e -= 1;
	}
	const scale = Math.max(0, WORK_DIGITS + 2 - Math.ceil(digitCount(n) / 2));
	const scaled = n * 10n ** BigInt(2 * scale);
	const root = integerSquareRoot(scaled);
	if (root * root !== scaled) precision.inexact = true;
	return roundSignificant(normalize(root, (e - 2 * scale) / 2), WORK_DIGITS, precision);
}

export function cubeRoot(value: Dec, precision: Precision): Dec {
	if (value.n === 0n) return ZERO;
	const negative = value.n < 0n;
	let n = abs(value.n);
	let e = value.e;
	while (((e % 3) + 3) % 3 !== 0) {
		n *= 10n;
		e -= 1;
	}
	const scale = Math.max(0, WORK_DIGITS + 2 - Math.ceil(digitCount(n) / 3));
	const scaled = n * 10n ** BigInt(3 * scale);
	const root = integerCubeRoot(scaled);
	if (root * root * root !== scaled) precision.inexact = true;
	const result = roundSignificant(normalize(root, (e - 3 * scale) / 3), WORK_DIGITS, precision);
	return negative ? negate(result) : result;
}

/** `base ^ exponent`. Integer exponents are exact up to WORK_DIGITS; fractional ones are double precision. */
export function power(base: Dec, exponent: Dec, precision: Precision): Dec {
	if (exponent.n === 0n) return ONE;
	if (base.n === 0n && exponent.n < 0n) throw new CalculatorError("Division by zero.");
	if (isInteger(exponent)) {
		if (digitCount(exponent.n) + exponent.e > String(MAX_INTEGER_EXPONENT).length) {
			throw new CalculatorError(`The exponent is too large (at most ${MAX_INTEGER_EXPONENT}).`);
		}
		const times = Number(abs(exponent.n) * 10n ** BigInt(exponent.e));
		if (times > MAX_INTEGER_EXPONENT) {
			throw new CalculatorError(`The exponent is too large (at most ${MAX_INTEGER_EXPONENT}).`);
		}
		let result = ONE;
		let square = base;
		for (let remaining = times; remaining > 0; remaining = Math.floor(remaining / 2)) {
			if (remaining % 2 === 1) result = multiply(result, square, precision);
			if (remaining > 1) square = multiply(square, square, precision);
		}
		return exponent.n < 0n ? divide(ONE, result, precision) : result;
	}
	if (base.n < 0n) throw new CalculatorError("A negative number cannot be raised to a fractional power.");
	if (base.n === 0n) return ZERO;
	if (exponent.n === 5n && exponent.e === -1) return squareRoot(base, precision);
	return fromDouble(toDouble(base) ** toDouble(exponent), precision, "pow");
}

/** Plain digits up to 10^40, scientific notation beyond (and below 1e-7), trailing zeros trimmed. */
export function formatDecimal(value: Dec): string {
	if (value.n === 0n) return "0";
	const negative = value.n < 0n;
	const digits = abs(value.n).toString();
	const adjusted = digits.length - 1 + value.e;
	let text: string;
	if (adjusted >= RESULT_DIGITS || adjusted < -7) {
		const rest = digits.slice(1);
		text = `${digits[0]}${rest === "" ? "" : `.${rest}`}e${adjusted < 0 ? "-" : "+"}${Math.abs(adjusted)}`;
	} else if (value.e >= 0) {
		text = digits + "0".repeat(value.e);
	} else if (-value.e >= digits.length) {
		text = `0.${"0".repeat(-value.e - digits.length)}${digits}`;
	} else {
		text = `${digits.slice(0, digits.length + value.e)}.${digits.slice(digits.length + value.e)}`;
	}
	return negative ? `-${text}` : text;
}
