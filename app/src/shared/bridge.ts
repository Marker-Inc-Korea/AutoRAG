import { APP_VERSION, type DevLabel } from "./app-info";

export interface AutoRagBridge {
	readonly version: string;
	/** Present only for unpackaged/dev runs: which clone this window runs from. */
	readonly dev: DevLabel | null;
}

export function createBridge(dev: DevLabel | null = null): AutoRagBridge {
	return { version: APP_VERSION, dev };
}
