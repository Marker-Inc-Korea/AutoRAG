import { contextBridge, ipcRenderer } from "electron";
import { createBridge } from "../shared/bridge";
import { createFsBridge } from "./fs-bridge";

contextBridge.exposeInMainWorld("autorag", { ...createBridge(), fs: createFsBridge(ipcRenderer) });
