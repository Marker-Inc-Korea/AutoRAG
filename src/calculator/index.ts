import { CalculatorError } from "./errors.ts";
import { evaluateExpression } from "./expression.ts";
import {
	addBusinessDays,
	businessDaysBetween,
	type CalculatorRequest,
	type Clock,
	convertTimeZone,
	dateAdd,
	dateDiff,
	dateInfo,
	type Fields,
	need,
	now,
} from "./operations.ts";
import { parseZone } from "./timezone.ts";

export { CalculatorError } from "./errors.ts";
export { MAX_EXPRESSION_LENGTH } from "./expression.ts";
export type { CalculatorRequest, Fields } from "./operations.ts";

export const CALCULATOR_OPERATIONS = [
	"evaluate",
	"now",
	"date_diff",
	"date_add",
	"date_info",
	"convert_timezone",
	"business_days_between",
	"add_business_days",
] as const;

export interface CalculatorOptions {
	/** The instant "now" and "today" mean. Defaults to the system clock. */
	readonly now?: Date;
	/** Zone used when the request names none. Defaults to the host's zone. */
	readonly hostTimeZone?: string;
}

export interface CalculatorOutput {
	readonly fields: Fields;
	/** One `name: value` line per field: the text the model reads. */
	readonly text: string;
}

const INEXACT_NOTE =
	"Rounded: arithmetic keeps 40 significant digits; exp, ln, log, trig and fractional powers are accurate to about 15.";

function evaluate(request: CalculatorRequest): Fields {
	const result = evaluateExpression(need("evaluate", "expression", request.expression));
	return {
		expression: result.expression,
		value: typeof result.value === "string" ? result.value : JSON.stringify(result.value),
		exact: result.exact,
		...(result.exact ? {} : { note: INEXACT_NOTE }),
	};
}

export function calculate(request: CalculatorRequest, options: CalculatorOptions = {}): CalculatorOutput {
	const requested = request.timeZone?.trim();
	const sourceGiven = requested !== undefined && requested !== "";
	const clock: Clock = {
		now: options.now ?? new Date(),
		zone: parseZone(
			sourceGiven ? requested : (options.hostTimeZone ?? Intl.DateTimeFormat().resolvedOptions().timeZone),
		),
		zoneSource: sourceGiven ? "requested" : "host default",
	};
	let fields: Fields;
	switch (request.operation) {
		case "evaluate":
			fields = evaluate(request);
			break;
		case "now":
			fields = now(clock);
			break;
		case "date_diff":
			fields = dateDiff(request, clock);
			break;
		case "date_add":
			fields = dateAdd(request, clock);
			break;
		case "date_info":
			fields = dateInfo(request, clock);
			break;
		case "convert_timezone":
			fields = convertTimeZone(request, clock, sourceGiven);
			break;
		case "business_days_between":
			fields = businessDaysBetween(request, clock);
			break;
		case "add_business_days":
			fields = addBusinessDays(request, clock);
			break;
		default:
			throw new CalculatorError(
				`Unknown operation "${request.operation}". Use one of: ${CALCULATOR_OPERATIONS.join(", ")}.`,
			);
	}
	return {
		fields,
		text: Object.entries(fields)
			.map(([name, value]) => `${name}: ${value}`)
			.join("\n"),
	};
}
