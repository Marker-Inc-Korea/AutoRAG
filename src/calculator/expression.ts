import {
	add,
	compare,
	cubeRoot,
	type Dec,
	digitCount,
	divide,
	formatDecimal,
	fromDouble,
	fromInteger,
	isInteger,
	multiply,
	negate,
	normalize,
	ONE,
	type Precision,
	parseDecimal,
	power,
	RESULT_DIGITS,
	remainder,
	roundAtExponent,
	roundSignificant,
	sign,
	squareRoot,
	subtract,
	sumExact,
	toDouble,
	ZERO,
} from "./decimal.ts";
import { CalculatorError } from "./errors.ts";

export const MAX_EXPRESSION_LENGTH = 4096;
/** Deepest chain of nested parentheses, lists, calls and unary operators. */
const MAX_DEPTH = 64;
const MAX_FACTORIAL = 1000;
/** Angles beyond this lose every digit in a double. */
const MAX_TRIG_INPUT = 1e15;

const PI = parseDecimal("3.14159265358979323846264338327950288419716939937510582097494");
const E = parseDecimal("2.71828182845904523536028747135266249775724709369995957496696");
const HUNDRED = fromInteger(100);
const HALF_TURN = fromInteger(180);

export interface ExpressionResult {
	readonly expression: string;
	readonly value: string | string[];
	/** False when any step rounded or approximated, so the digits are the best 40-digit answer rather than the exact one. */
	readonly exact: boolean;
}

type Value = Dec | Dec[];

interface Token {
	readonly kind: "number" | "name" | "op" | "end";
	readonly text: string;
}

const UNICODE_OPERATORS: Readonly<Record<string, string>> = { "×": "*", "÷": "/", "−": "-", "–": "-", "∗": "*" };

function tokenize(source: string): Token[] {
	const tokens: Token[] = [];
	let index = 0;
	while (index < source.length) {
		const rest = source.slice(index);
		const space = /^\s+/u.exec(rest);
		if (space !== null) {
			index += space[0].length;
			continue;
		}
		const number = /^(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/u.exec(rest);
		if (number !== null) {
			tokens.push({ kind: "number", text: number[0] });
			index += number[0].length;
			continue;
		}
		const name = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(rest);
		if (name !== null) {
			tokens.push({ kind: "name", text: name[0] });
			index += name[0].length;
			continue;
		}
		const char = rest[0] as string;
		if (rest.startsWith("**")) {
			tokens.push({ kind: "op", text: "^" });
			index += 2;
		} else if ("+-*/%^!()[],".includes(char)) {
			tokens.push({ kind: "op", text: char });
			index += 1;
		} else if (char in UNICODE_OPERATORS) {
			tokens.push({ kind: "op", text: UNICODE_OPERATORS[char] as string });
			index += 1;
		} else {
			throw new CalculatorError(`Unexpected character "${char}".`);
		}
	}
	tokens.push({ kind: "end", text: "" });
	return tokens;
}

function scalar(value: Value, what: string): Dec {
	if (Array.isArray(value)) throw new CalculatorError(`${what} expects a single number, not a list.`);
	return value;
}

/** Every number among the arguments; a list argument contributes its members. */
function members(args: readonly Value[], name: string, minimum = 1): Dec[] {
	const values = args.flatMap((arg) => (Array.isArray(arg) ? arg : [arg]));
	if (values.length < minimum) {
		throw new CalculatorError(
			`${name} needs at least ${minimum === 1 ? "one" : "two"} value${minimum === 1 ? "" : "s"}.`,
		);
	}
	return values;
}

function requireInteger(value: Dec, name: string, low: number, high: number): number {
	if (!isInteger(value) || compare(value, fromInteger(low)) < 0 || compare(value, fromInteger(high)) > 0) {
		throw new CalculatorError(`${name} needs a whole number from ${low} to ${high}.`);
	}
	return Number(value.n * 10n ** BigInt(value.e));
}

function factorial(value: Dec, precision: Precision): Dec {
	if (!isInteger(value) || sign(value) < 0) throw new CalculatorError("Factorial needs a non-negative integer.");
	if (compare(value, fromInteger(MAX_FACTORIAL)) > 0) {
		throw new CalculatorError(`Factorial supports operands up to ${MAX_FACTORIAL}.`);
	}
	let result = ONE;
	for (let factor = 2; factor <= Number(value.n * 10n ** BigInt(value.e)); factor++) {
		result = multiply(result, fromInteger(factor), precision);
	}
	return result;
}

/** Mantissa in [1, 10) and decimal exponent, so logarithms never overflow a double. */
function mantissaAndExponent(value: Dec): { mantissa: number; exponent: number } {
	const digits = (value.n < 0n ? -value.n : value.n).toString();
	return { mantissa: Number(`${digits[0]}.${digits.slice(1, 21)}`), exponent: digits.length - 1 + value.e };
}

function requirePositive(value: Dec, name: string): void {
	if (sign(value) <= 0) throw new CalculatorError(`${name} needs a positive number.`);
}

/** Natural, base-10 or base-2 logarithm. A value that fits a double is logged directly; the mantissa split only serves huge or tiny ones, where it would otherwise overflow. */
function logarithm(value: Dec, name: string, base: "e" | 10 | 2, precision: Precision): Dec {
	requirePositive(value, name);
	const direct = toDouble(value);
	if (Number.isFinite(direct) && direct > 0) {
		return fromDouble(
			base === "e" ? Math.log(direct) : base === 10 ? Math.log10(direct) : Math.log2(direct),
			precision,
			name,
		);
	}
	const { mantissa, exponent } = mantissaAndExponent(value);
	const result =
		base === "e"
			? Math.log(mantissa) + exponent * Math.LN10
			: base === 10
				? Math.log10(mantissa) + exponent
				: (Math.log10(mantissa) + exponent) / Math.log10(2);
	return fromDouble(result, precision, name);
}

function angle(value: Dec, name: string): number {
	const radians = toDouble(value);
	if (!Number.isFinite(radians) || Math.abs(radians) > MAX_TRIG_INPUT) {
		throw new CalculatorError(`${name} needs an angle below ${MAX_TRIG_INPUT} radians.`);
	}
	return radians;
}

/** An angle this close to a whole number of quarter turns is that multiple: the input was a rounded π, not a real offset. */
const QUARTER_TURN_TOLERANCE = parseDecimal("1e-30");

function abs(value: Dec): Dec {
	return sign(value) < 0 ? negate(value) : value;
}

/**
 * sin, cos and tan. At whole quarter turns (`exact` indexed by quarter turns mod 4, `undefined` where the
 * function does not exist) the answer is exact; a double cannot hold π, so Math.sin(π) is 1.2e-16, not 0.
 */
function trigFunction(
	name: string,
	run: (x: number) => number,
	exact: readonly [number, number | undefined, number, number | undefined],
): FunctionSpec["run"] {
	return (args, { precision }) => {
		const value = scalar(args[0] as Value, name);
		const radians = angle(value, name);
		const quarterTurns = divide(value, divide(PI, fromInteger(2), precision), precision);
		const nearest = roundAtExponent(quarterTurns, 0, "half-away", { inexact: false });
		if (compare(abs(subtract(quarterTurns, nearest, precision)), QUARTER_TURN_TOLERANCE) < 0) {
			const turns = Number((nearest.n * 10n ** BigInt(nearest.e)) % 4n);
			const known = exact[(turns + 4) % 4];
			if (known === undefined) throw new CalculatorError(`${name} is undefined at ${formatDecimal(value)} radians.`);
			return fromInteger(known);
		}
		return fromDouble(run(radians), precision, name);
	};
}

interface Context {
	readonly precision: Precision;
}

interface FunctionSpec {
	readonly min: number;
	readonly max: number;
	readonly run: (args: readonly Value[], context: Context) => Dec;
}

const doubleFunction =
	(name: string, run: (x: number) => number, input: (value: Dec) => number = toDouble): FunctionSpec["run"] =>
	(args, { precision }) =>
		fromDouble(run(input(scalar(args[0] as Value, name))), precision, name);

/** Rounding functions give the defined result, so they never mark the answer inexact. */
function roundedTo(value: Dec, exponent: number, mode: "half-away" | "floor" | "ceil" | "trunc"): Dec {
	return roundAtExponent(value, exponent, mode, { inexact: false });
}

const FUNCTIONS: ReadonlyMap<string, FunctionSpec> = new Map<string, FunctionSpec>([
	["abs", { min: 1, max: 1, run: ([x]) => (sign(scalar(x as Value, "abs")) < 0 ? negate(x as Dec) : (x as Dec)) }],
	["sign", { min: 1, max: 1, run: ([x]) => fromInteger(sign(scalar(x as Value, "sign"))) }],
	["sqrt", { min: 1, max: 1, run: ([x], { precision }) => squareRoot(scalar(x as Value, "sqrt"), precision) }],
	["cbrt", { min: 1, max: 1, run: ([x], { precision }) => cubeRoot(scalar(x as Value, "cbrt"), precision) }],
	[
		"pow",
		{
			min: 2,
			max: 2,
			run: ([x, y], { precision }) => power(scalar(x as Value, "pow"), scalar(y as Value, "pow"), precision),
		},
	],
	["exp", { min: 1, max: 1, run: doubleFunction("exp", Math.exp) }],
	["ln", { min: 1, max: 1, run: ([x], { precision }) => logarithm(scalar(x as Value, "ln"), "ln", "e", precision) }],
	[
		"log10",
		{ min: 1, max: 1, run: ([x], { precision }) => logarithm(scalar(x as Value, "log10"), "log10", 10, precision) },
	],
	[
		"log2",
		{ min: 1, max: 1, run: ([x], { precision }) => logarithm(scalar(x as Value, "log2"), "log2", 2, precision) },
	],
	[
		"log",
		{
			min: 1,
			max: 2,
			run: (args, { precision }) => {
				if (args.length === 1) {
					throw new CalculatorError("log(x) is ambiguous: use ln(x), log10(x) or log(x, base).");
				}
				const value = logarithm(scalar(args[0] as Value, "log"), "log", 10, precision);
				const base = logarithm(scalar(args[1] as Value, "log"), "log", 10, precision);
				if (base.n === 0n) throw new CalculatorError("log base must not be 1.");
				return fromDouble(toDouble(value) / toDouble(base), precision, "log");
			},
		},
	],
	["sin", { min: 1, max: 1, run: trigFunction("sin", Math.sin, [0, 1, 0, -1]) }],
	["cos", { min: 1, max: 1, run: trigFunction("cos", Math.cos, [1, 0, -1, 0]) }],
	["tan", { min: 1, max: 1, run: trigFunction("tan", Math.tan, [0, undefined, 0, undefined]) }],
	["asin", { min: 1, max: 1, run: doubleFunction("asin", Math.asin) }],
	["acos", { min: 1, max: 1, run: doubleFunction("acos", Math.acos) }],
	["atan", { min: 1, max: 1, run: doubleFunction("atan", Math.atan) }],
	[
		"atan2",
		{
			min: 2,
			max: 2,
			run: ([y, x], { precision }) =>
				fromDouble(
					Math.atan2(toDouble(scalar(y as Value, "atan2")), toDouble(scalar(x as Value, "atan2"))),
					precision,
					"atan2",
				),
		},
	],
	["sinh", { min: 1, max: 1, run: doubleFunction("sinh", Math.sinh) }],
	["cosh", { min: 1, max: 1, run: doubleFunction("cosh", Math.cosh) }],
	["tanh", { min: 1, max: 1, run: doubleFunction("tanh", Math.tanh) }],
	[
		"radians",
		{
			min: 1,
			max: 1,
			run: ([x], { precision }) =>
				divide(multiply(scalar(x as Value, "radians"), PI, precision), HALF_TURN, precision),
		},
	],
	[
		"degrees",
		{
			min: 1,
			max: 1,
			run: ([x], { precision }) =>
				divide(multiply(scalar(x as Value, "degrees"), HALF_TURN, precision), PI, precision),
		},
	],
	["round", { min: 1, max: 1, run: ([x]) => roundedTo(scalar(x as Value, "round"), 0, "half-away") }],
	["floor", { min: 1, max: 1, run: ([x]) => roundedTo(scalar(x as Value, "floor"), 0, "floor") }],
	["ceil", { min: 1, max: 1, run: ([x]) => roundedTo(scalar(x as Value, "ceil"), 0, "ceil") }],
	["trunc", { min: 1, max: 1, run: ([x]) => roundedTo(scalar(x as Value, "trunc"), 0, "trunc") }],
	[
		"roundto",
		{
			min: 2,
			max: 2,
			run: ([x, places]) =>
				roundedTo(
					scalar(x as Value, "roundTo"),
					-requireInteger(scalar(places as Value, "roundTo places"), "roundTo places", -30, 30),
					"half-away",
				),
		},
	],
	[
		"percent",
		{
			min: 2,
			max: 2,
			run: ([rate, amount], { precision }) =>
				divide(
					multiply(scalar(rate as Value, "percent"), scalar(amount as Value, "percent"), precision),
					HUNDRED,
					precision,
				),
		},
	],
	[
		"pctchange",
		{
			min: 2,
			max: 2,
			run: ([from, to], { precision }) => {
				const start = scalar(from as Value, "pctChange");
				const change = subtract(scalar(to as Value, "pctChange"), start, precision);
				return multiply(divide(change, start, precision), HUNDRED, precision);
			},
		},
	],
	["fac", { min: 1, max: 1, run: ([x], { precision }) => factorial(scalar(x as Value, "fac"), precision) }],
	[
		"sum",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args, { precision }) => sumExact(members(args, "sum"), precision),
		},
	],
	[
		"mean",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args, { precision }) => {
				const values = members(args, "mean");
				return divide(sumExact(values, precision), fromInteger(values.length), precision);
			},
		},
	],
	[
		"median",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args, { precision }) => {
				const sorted = members(args, "median").sort(compare);
				const middle = sorted.length >> 1;
				return sorted.length % 2 === 1
					? (sorted[middle] as Dec)
					: divide(add(sorted[middle - 1] as Dec, sorted[middle] as Dec, precision), fromInteger(2), precision);
			},
		},
	],
	[
		"min",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args) => members(args, "min").reduce((a, b) => (compare(b, a) < 0 ? b : a)),
		},
	],
	[
		"max",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args) => members(args, "max").reduce((a, b) => (compare(b, a) > 0 ? b : a)),
		},
	],
	[
		"stdev",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args, { precision }) => deviation(members(args, "stdev"), 0, precision),
		},
	],
	[
		"stdevs",
		{
			min: 1,
			max: Number.POSITIVE_INFINITY,
			run: (args, { precision }) => deviation(members(args, "stdevs", 2), 1, precision),
		},
	],
]);

/** Standard deviation; `freedom` 0 divides by n (population), 1 by n - 1 (sample). */
function deviation(values: readonly Dec[], freedom: 0 | 1, precision: Precision): Dec {
	const mean = divide(sumExact(values, precision), fromInteger(values.length), precision);
	const squares = values.map((value) => {
		const delta = subtract(value, mean, precision);
		return multiply(delta, delta, precision);
	});
	return squareRoot(divide(sumExact(squares, precision), fromInteger(values.length - freedom), precision), precision);
}

/** Maps, not records: keys are model-typed, and `constructor` or `__proto__` must not resolve through the prototype. */
const CONSTANTS: ReadonlyMap<string, Dec> = new Map([
	["pi", PI],
	["e", E],
]);

class Parser {
	private position = 0;
	private depth = 0;
	readonly precision: Precision = { inexact: false };

	private readonly tokens: readonly Token[];

	constructor(tokens: readonly Token[]) {
		this.tokens = tokens;
	}

	parse(): Value {
		const value = this.parseExpression();
		const next = this.peek();
		if (next.kind !== "end") {
			if (next.text === ",") {
				throw new CalculatorError(
					'Unexpected ",". Write numbers without thousands separators (1234, not 1,234); use [a, b] for several results.',
				);
			}
			throw new CalculatorError(`Unexpected "${next.text}".`);
		}
		return value;
	}

	private peek(): Token {
		return this.tokens[this.position] as Token;
	}

	private take(): Token {
		return this.tokens[this.position++] as Token;
	}

	private accept(...operators: string[]): string | undefined {
		const token = this.peek();
		if (token.kind === "op" && operators.includes(token.text)) {
			this.position++;
			return token.text;
		}
		return undefined;
	}

	private expect(operator: string): void {
		const token = this.take();
		if (token.kind === "end") throw new CalculatorError(`Expected "${operator}" but the expression ended.`);
		if (token.text !== operator) throw new CalculatorError(`Expected "${operator}" but found "${token.text}".`);
	}

	private enter(): void {
		if (++this.depth > MAX_DEPTH)
			throw new CalculatorError(`Expression is nested too deeply (at most ${MAX_DEPTH} levels).`);
	}

	private parseExpression(): Value {
		const first = this.parseTerm();
		if (!(this.peek().kind === "op" && "+-".includes(this.peek().text))) return first;
		let result = scalar(first, "Operator +/-");
		for (let operator = this.accept("+", "-"); operator !== undefined; operator = this.accept("+", "-")) {
			const right = scalar(this.parseTerm(), "Operator +/-");
			result = operator === "+" ? add(result, right, this.precision) : subtract(result, right, this.precision);
		}
		return result;
	}

	private parseTerm(): Value {
		const first = this.parseUnary();
		if (!(this.peek().kind === "op" && "*/%".includes(this.peek().text))) return first;
		let result = scalar(first, "Operator * / %");
		for (let operator = this.accept("*", "/", "%"); operator !== undefined; operator = this.accept("*", "/", "%")) {
			const right = scalar(this.parseUnary(), "Operator * / %");
			result =
				operator === "*"
					? multiply(result, right, this.precision)
					: operator === "/"
						? divide(result, right, this.precision)
						: remainder(result, right, this.precision);
		}
		return result;
	}

	/** Unary minus binds looser than `^`, so -2^2 is -4. */
	private parseUnary(): Value {
		this.enter();
		try {
			const operator = this.accept("+", "-");
			if (operator === undefined) return this.parsePower();
			const operand = scalar(this.parseUnary(), `Unary ${operator}`);
			return operator === "-" ? negate(operand) : operand;
		} finally {
			this.depth--;
		}
	}

	/** `^` is right-associative and its exponent may carry a sign: 2^3^2 is 512, 2^-2 is 0.25. */
	private parsePower(): Value {
		const base = this.parsePostfix();
		if (this.accept("^") === undefined) return base;
		return power(scalar(base, "Operator ^"), scalar(this.parseUnary(), "Operator ^"), this.precision);
	}

	private parsePostfix(): Value {
		let value = this.parsePrimary();
		while (this.accept("!") !== undefined) value = factorial(scalar(value, "Factorial"), this.precision);
		return value;
	}

	private parsePrimary(): Value {
		const token = this.take();
		if (token.kind === "end") throw new CalculatorError("Unexpected end of expression.");
		if (token.kind === "number") return parseDecimal(token.text);
		if (token.kind === "name") return this.parseName(token.text);
		if (token.text === "(") {
			this.enter();
			const inner = this.parseExpression();
			this.expect(")");
			this.depth--;
			return inner;
		}
		if (token.text === "[") return this.parseList();
		throw new CalculatorError(`Unexpected "${token.text}".`);
	}

	private parseName(name: string): Value {
		const key = name.toLowerCase();
		if (this.peek().text !== "(" || this.peek().kind !== "op") {
			const constant = CONSTANTS.get(key);
			if (constant === undefined) throw new CalculatorError(`Unknown name "${name}".`);
			return constant;
		}
		const spec = FUNCTIONS.get(key);
		if (spec === undefined) throw new CalculatorError(`Unknown function "${name}".`);
		this.take();
		this.enter();
		const args: Value[] = [];
		if (this.accept(")") === undefined) {
			do args.push(this.parseExpression());
			while (this.accept(",") !== undefined);
			this.expect(")");
		}
		this.depth--;
		if (args.length < spec.min || args.length > spec.max) {
			const wanted =
				spec.max === Number.POSITIVE_INFINITY
					? `at least ${spec.min}`
					: spec.min === spec.max
						? `${spec.min}`
						: `${spec.min} to ${spec.max}`;
			throw new CalculatorError(`${name}() expects ${wanted} argument(s) but got ${args.length}.`);
		}
		return spec.run(args, { precision: this.precision });
	}

	private parseList(): Dec[] {
		this.enter();
		const items: Dec[] = [];
		if (this.accept("]") === undefined) {
			do {
				const item = this.parseExpression();
				if (Array.isArray(item)) throw new CalculatorError("Nested lists are not supported; use one flat list.");
				items.push(item);
			} while (this.accept(",") !== undefined);
			this.expect("]");
		}
		this.depth--;
		return items;
	}
}

function finalize(value: Dec, precision: Precision): string {
	return formatDecimal(roundSignificant(value, RESULT_DIGITS, precision));
}

/** Evaluates one expression, or one flat `[a, b, …]` list of expressions, with exact decimal arithmetic. */
export function evaluateExpression(source: string): ExpressionResult {
	const expression = source.trim();
	if (expression === "") throw new CalculatorError("Expression is empty.");
	if (expression.length > MAX_EXPRESSION_LENGTH) {
		throw new CalculatorError(`Expression must be at most ${MAX_EXPRESSION_LENGTH} characters.`);
	}
	const parser = new Parser(tokenize(expression));
	const result = parser.parse();
	if (Array.isArray(result) && result.length === 0) throw new CalculatorError("The list is empty.");
	const value = Array.isArray(result)
		? result.map((item) => finalize(item, parser.precision))
		: finalize(result, parser.precision);
	return { expression, value, exact: !parser.precision.inexact };
}

export { digitCount, normalize, ZERO };
