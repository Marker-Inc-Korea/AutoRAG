import { contextBridge, ipcRenderer } from "electron";
import { createBridge } from "../shared/bridge";
import { createFsBridge } from "./fs-bridge";
import { createSearchBridge } from "./search-bridge";
import { createSettingsBridge } from "./settings-bridge";

contextBridge.exposeInMainWorld("autorag", {
	...createBridge(),
	fs: createFsBridge(ipcRenderer),
	search: createSearchBridge(ipcRenderer),
	settings: createSettingsBridge(ipcRenderer),
});
