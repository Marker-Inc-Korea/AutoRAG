import type { AutoRagBridge } from "./bridge";
import type { FsBridge } from "./fs-contract";
import type { SearchBridge } from "./search-contract";
import type { SettingsBridge } from "./settings-contract";

declare global {
	interface Window {
		readonly autorag: AutoRagBridge & {
			readonly fs: FsBridge;
			readonly search: SearchBridge;
			readonly settings: SettingsBridge;
		};
	}
}
