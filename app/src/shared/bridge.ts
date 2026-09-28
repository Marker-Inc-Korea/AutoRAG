import { APP_VERSION } from "./app-info";

export interface AutoRagBridge {
	readonly version: string;
}

export function createBridge(): AutoRagBridge {
	return { version: APP_VERSION };
}
