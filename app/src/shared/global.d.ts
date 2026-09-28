import type { AutoRagBridge } from "./bridge";

declare global {
	interface Window {
		readonly autorag: AutoRagBridge;
	}
}
