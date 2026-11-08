/** A calculation the calculator refuses: bad input or a result it cannot give honestly. The message is for the model. */
export class CalculatorError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "CalculatorError";
	}
}
