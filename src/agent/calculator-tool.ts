import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import {
	CALCULATOR_OPERATIONS,
	CalculatorError,
	type CalculatorRequest,
	calculate,
	MAX_EXPRESSION_LENGTH,
} from "../calculator/index.ts";

export const CALCULATOR_TOOL_NAME = "calculator";

/** The standing order every prompt repeats: calculations belong to the tool, never to the model. */
export const CALCULATOR_PROMPT_RULE = `Never do arithmetic, date, time, or time zone math yourself: use ${CALCULATOR_TOOL_NAME} for every sum, percentage, average, unit conversion, day count, date offset, weekday, and "what time is it in…", and copy its result unchanged. The current date in your instructions is a starting point, not a substitute for the tool.`;

/** The fast phase forbids tools; this carves out the one that reads no sources and cannot slow the answer down. */
export const CALCULATOR_FAST_PHASE_RULE = `${CALCULATOR_TOOL_NAME} is allowed: ${CALCULATOR_PROMPT_RULE}`;

export interface CalculatorToolOptions {
	/** The instant "now" and "today" mean. Tests pin it; the default reads the system clock on every call. */
	readonly now?: () => Date;
	/** Zone used when a call names none. Defaults to the host's zone. */
	readonly hostTimeZone?: string;
}

const operation = Type.Enum(CALCULATOR_OPERATIONS, {
	description:
		"evaluate: arithmetic expression. now: current date/time in a zone. date_diff: days/time between two dates or times. date_add: add or subtract years/months/weeks/days/hours/minutes/seconds. date_info: weekday, day of year, ISO week, leap year. convert_timezone: one instant in another zone. business_days_between: weekdays between two dates. add_business_days: move a date by weekdays.",
});

const count = (description: string) => Type.Optional(Type.Integer({ description }));

const calculatorSchema = Type.Object({
	operation,
	expression: Type.Optional(
		Type.String({
			maxLength: MAX_EXPRESSION_LENGTH,
			description:
				'evaluate only. + - * / % ^ and parentheses, n!, PI, E, plain numbers (no thousands separators: 1234, not 1,234). Functions: sqrt cbrt abs sign pow exp ln log10 log2 log(x, base) sin cos tan asin acos atan atan2 sinh cosh tanh radians degrees round floor ceil trunc roundTo(x, places) percent(rate, amount) pctChange(from, to) fac sum mean median min max stdev stdevs (the last group take a list: sum([1, 2.5, 3])). Put independent calculations in one flat list: [0.1 + 0.2, 2^10]. Arithmetic is exact to 40 significant digits; trig and logs are accurate to about 15. Result "exact: false" means it was rounded.',
		}),
	),
	timeZone: Type.Optional(
		Type.String({
			description:
				"Zone a date or time is read in and shown in; for convert_timezone, the SOURCE zone. IANA name (Asia/Seoul, America/New_York), UTC, KST, JST, or a fixed offset (+09:00). Omit to use the host default, which the result states. Ambiguous abbreviations (PST, CST, IST) are refused.",
		}),
	),
	toTimeZone: Type.Optional(Type.String({ description: "convert_timezone only: the zone to convert into." })),
	date: Type.Optional(
		Type.String({
			description:
				'date_add, date_info, add_business_days: a date or time. YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD, "2026년 11월 8일", YYYY-MM-DDTHH:MM:SS[±HH:MM|Z], epoch seconds, "today" or "now".',
		}),
	),
	datetime: Type.Optional(
		Type.String({
			description: "convert_timezone only: the time to convert. Carry an offset or Z, or give timeZone.",
		}),
	),
	from: Type.Optional(Type.String({ description: "date_diff, business_days_between: start. Same formats as date." })),
	to: Type.Optional(Type.String({ description: "date_diff, business_days_between: end. Same formats as date." })),
	years: count("date_add: whole years to add (negative subtracts)."),
	months: count("date_add: whole months. A day past the end of the target month clamps to its last day."),
	weeks: count("date_add: whole weeks."),
	days: count("date_add: whole days; add_business_days: weekdays to move (negative goes back)."),
	hours: count("date_add: whole hours of elapsed time."),
	minutes: count("date_add: whole minutes of elapsed time."),
	seconds: count("date_add: whole seconds of elapsed time."),
	holidays: Type.Optional(
		Type.Array(Type.String(), {
			description:
				"business_days_between, add_business_days: dates to treat as non-working days besides Saturday and Sunday.",
		}),
	),
	inclusive: Type.Optional(
		Type.Boolean({
			description:
				"business_days_between: also count the later date. Default counts the start up to, not including, the end.",
		}),
	),
});

export type CalculatorDetails = Record<string, string | number | boolean>;

/**
 * The model's calculator: exact decimal arithmetic and calendar/clock math. Language models miscount days,
 * drop digits and forget daylight saving, so every numeric or date answer comes from this tool, not from the model.
 */
export function createCalculatorTool(
	options: CalculatorToolOptions = {},
): AgentTool<typeof calculatorSchema, CalculatorDetails> {
	return {
		name: CALCULATOR_TOOL_NAME,
		label: "Calculator",
		description:
			"Exact calculator for numbers, dates, times and time zones. Use it for every calculation whose answer you would otherwise work out yourself: arithmetic, percentages, totals, averages, unit math, days between dates, a date plus or minus a duration, weekday of a date, business days, the current date or time, and time zone conversion. Never do these in your head. Dates: date_diff, date_add, date_info, business_days_between, add_business_days. Time: now, convert_timezone. Numbers: evaluate. Several independent numbers go in one evaluate call as a list. Results are exact; copy them into the answer unchanged.",
		parameters: calculatorSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<CalculatorDetails>> {
			const request: CalculatorRequest = params;
			try {
				const { fields, text } = calculate(request, {
					...(options.now === undefined ? {} : { now: options.now() }),
					...(options.hostTimeZone === undefined ? {} : { hostTimeZone: options.hostTimeZone }),
				});
				return { content: [{ type: "text", text }], details: fields };
			} catch (error) {
				if (!(error instanceof CalculatorError)) throw error;
				return { content: [{ type: "text", text: error.message }], details: {}, isError: true };
			}
		},
	};
}
