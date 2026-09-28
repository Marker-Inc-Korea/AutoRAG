import type { AutoRagBridge } from "./bridge";
import type { FsBridge } from "./fs-contract";

declare global {
	interface Window {
		readonly autorag: AutoRagBridge & { readonly fs: FsBridge };
	}
}
