export class NotImplementedError extends Error {
	readonly methodName: string;

	constructor(methodName: string) {
		super(`[NotImplemented] ${methodName} retrieval is not yet implemented. This is a stub for future integration.`);
		this.name = "NotImplementedError";
		this.methodName = methodName;
	}
}
