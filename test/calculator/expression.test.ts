import { describe, expect, it } from "vitest";
import { CalculatorError } from "../../src/calculator/errors.ts";
import { evaluateExpression, MAX_EXPRESSION_LENGTH } from "../../src/calculator/expression.ts";

const value = (expression: string) => evaluateExpression(expression).value;

describe("calculator expressions: exact decimal arithmetic", () => {
	it.each([
		["0.1 + 0.2", "0.3"],
		["2^64", "18446744073709551616"],
		["123456789 * 987654321", "121932631112635269"],
		["(12.5 * 1.0825) ^ 3", "2477.500518798828125"],
		["1.05^10", "1.62889462677744140625"],
		["1/3", "0.3333333333333333333333333333333333333333"],
		["1/3*3", "1"],
		["10 - 0.0000001", "9.9999999"],
		["1e3 + 1.5e-3", "1000.0015"],
		["fac(25)", "15511210043330985984000000"],
		["5!", "120"],
		["2^-2", "0.25"],
		["-2^2", "-4"],
		["2^3^2", "512"],
		["2 ** 10", "1024"],
	])("%s = %s", (expression, expected) => {
		expect(value(expression)).toBe(expected);
	});

	it("keeps 40 significant digits of an irrational root and trims trailing zeros", () => {
		expect(value("sqrt(2)")).toBe("1.41421356237309504880168872420969807857");
		expect(value("sqrt(16)")).toBe("4");
		expect(value("sqrt(0.25)")).toBe("0.5");
		expect(value("cbrt(27)")).toBe("3");
	});

	it("uses a remainder whose sign follows the dividend", () => {
		expect(value("10 % 3")).toBe("1");
		expect(value("-7 % 3")).toBe("-1");
		expect(value("5.5 % 2")).toBe("1.5");
	});

	it("normalizes unicode operators a model or a user may paste", () => {
		expect(value("6 × 7")).toBe("42");
		expect(value("9 ÷ 3")).toBe("3");
		expect(value("5 − 8")).toBe("-3");
	});

	it("evaluates independent results in one flat array", () => {
		expect(value("[0.1 + 0.2, 2^10, 7/2]")).toEqual(["0.3", "1024", "3.5"]);
	});
});

describe("calculator expressions: functions", () => {
	it.each([
		["percent(15, 200)", "30"],
		["pctChange(80, 100)", "25"],
		["pctChange(100, 80)", "-20"],
		["round(2.5)", "3"],
		["round(-2.5)", "-3"],
		["roundTo(-1.25, 1)", "-1.3"],
		["roundTo(1234.5678, 2)", "1234.57"],
		["roundTo(1250, -2)", "1300"],
		["floor(-1.1)", "-2"],
		["ceil(1.1)", "2"],
		["trunc(-1.9)", "-1"],
		["abs(-3.5)", "3.5"],
		["sign(-9)", "-1"],
		["min(3, 1, 2)", "1"],
		["max([3, 1, 2], 10)", "10"],
		["sum([1, 2, 3.5])", "6.5"],
		["sum([1e40, 1, -1e40])", "1"],
		["mean([2, 4, 6, 8])", "5"],
		["median([5, 1, 3])", "3"],
		["median([1, 2, 3, 4])", "2.5"],
		["stdev([2,4,4,4,5,5,7,9])", "2"],
		["stdevs([2,4,4,4,5,5,7,9])", "2.138089935299395077476427847038028172432"],
		["pow(4, 0.5)", "2"],
		["log10(1000)", "3"],
		["log2(1024)", "10"],
		["log(8, 2)", "3"],
		["ln(1)", "0"],
		["sin(radians(90))", "1"],
		["degrees(PI)", "180"],
		["PI", "3.141592653589793238462643383279502884197"],
		["sin(radians(180))", "0"],
		["sin(radians(30))", "0.5"],
		["sin(radians(270))", "-1"],
		["cos(radians(90))", "0"],
		["cos(radians(60))", "0.5"],
		["cos(PI)", "-1"],
		["tan(radians(45))", "1"],
		["tan(radians(135))", "-1"],
		["tan(radians(180))", "0"],
	])("%s = %s", (expression, expected) => {
		expect(value(expression)).toBe(expected);
	});

	it("treats function names and constants case-insensitively", () => {
		expect(value("ROUNDTO(1.005, 2) + pi - PI")).toBe("1.01");
	});

	it("rounds transcendental results to 15 significant digits (double precision)", () => {
		expect(value("exp(1)")).toBe("2.71828182845905");
		expect(value("ln(2)")).toBe("0.693147180559945");
		expect(value("sin(-1)")).toBe("-0.841470984807897");
		expect(value("ln(0.5)")).toBe("-0.693147180559945");
		expect(value("exp(-1)")).toBe("0.367879441171442");
		expect(value("log10(1e500)")).toBe("500");
		expect(value("log10(1e-500)")).toBe("-500");
		expect(value("ln(1e400)")).toBe("921.034037197618");
	});
});

describe("calculator expressions: rejects what it cannot compute exactly or safely", () => {
	it.each([
		["", /empty/i],
		["   ", /empty/i],
		["2 +", /unexpected end/i],
		["(1 + 2", /\)/],
		["1 +* 2", /unexpected/i],
		["5 / 0", /division by zero/i],
		["5 % 0", /division by zero/i],
		["0^-1", /division by zero/i],
		["sqrt(-1)", /negative/i],
		["ln(0)", /positive/i],
		["log(8)", /ambiguous.*ln\(x\).*log10\(x\).*log\(x, base\)/i],
		["fac(-1)", /non-negative integer/i],
		["fac(1001)", /1000/],
		["fac(2.5)", /non-negative integer/i],
		["foo(1)", /unknown function "foo"/i],
		["bar + 1", /unknown name "bar"/i],
		["tan(radians(90))", /undefined/i],
		["tan(PI / 2)", /undefined/i],
		["__proto__", /unknown name/i],
		["constructor(1)", /unknown function/i],
		["sqrt()", /argument/i],
		["round(1, 2, 3)", /argument/i],
		["mean([])", /at least one/i],
		["stdevs([1])", /at least two/i],
		["(-8)^0.5", /negative/i],
		["2^100000", /too large/i],
		["1,234 + 1", /thousands separator/i],
		["1 $ 2", /unexpected character "\$"/i],
		["[[1], 2]", /nested/i],
		["sum(1, [2, [3]])", /nested/i],
		["sin([1, 2])", /single number/i],
	])("%j fails with %s", (expression, message) => {
		expect(() => evaluateExpression(expression)).toThrow(CalculatorError);
		expect(() => evaluateExpression(expression)).toThrow(message);
	});

	it("caps expression length and nesting depth", () => {
		expect(() => evaluateExpression("1+".repeat(MAX_EXPRESSION_LENGTH))).toThrow(/at most/i);
		expect(() => evaluateExpression(`${"(".repeat(200)}1${")".repeat(200)}`)).toThrow(/nest/i);
	});
});

describe("calculator expressions: result shape", () => {
	it("echoes the trimmed expression and a string value", () => {
		expect(evaluateExpression("  1 + 1  ")).toEqual({ expression: "1 + 1", value: "2", exact: true });
	});

	it("flags a result that was rounded to 40 significant digits as inexact", () => {
		expect(evaluateExpression("1/3").exact).toBe(false);
		expect(evaluateExpression("1/4").exact).toBe(true);
		expect(evaluateExpression("[1/4, 1/3]").exact).toBe(false);
	});

	it("switches to scientific notation outside the plain range", () => {
		expect(value("1e50 / 3")).toBe("3.333333333333333333333333333333333333333e+49");
		expect(value("1 / 3e30")).toBe("3.333333333333333333333333333333333333333e-31");
		expect(value("2^400")).toBe("2.58224987808690858965591917200301187433e+120");
	});
});
