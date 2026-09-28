import { contextBridge } from "electron";
import { createBridge } from "../shared/bridge";

contextBridge.exposeInMainWorld("autorag", createBridge());
