import { contextBridge, ipcRenderer } from "electron";
import { createBridge } from "../shared/bridge";
import type { DevLabel } from "../shared/app-info";
import { createFsBridge } from "./fs-bridge";
import { createSearchBridge } from "./search-bridge";
import { createSettingsBridge } from "./settings-bridge";

const DEV_LABEL_FLAG = "--autorag-dev-label=";

function readDevLabelArg(): DevLabel | null {
	const arg = process.argv.find((value) => value.startsWith(DEV_LABEL_FLAG));
	if (arg === undefined) return null;
	try {
		return JSON.parse(arg.slice(DEV_LABEL_FLAG.length)) as DevLabel;
	} catch {
		return null;
	}
}

contextBridge.exposeInMainWorld("autorag", {
	...createBridge(readDevLabelArg()),
	fs: createFsBridge(ipcRenderer),
	search: createSearchBridge(ipcRenderer),
	settings: createSettingsBridge(ipcRenderer),
});
