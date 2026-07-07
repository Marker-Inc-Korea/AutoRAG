export { JikjiClient } from "./client.ts";
export type { JikjiDiagnostic, JikjiDiagnosticCode } from "./diagnostics.ts";
export { jikjiPrepareDiagnostic } from "./diagnostics.ts";
export type { JikjiFileMapEntry, JikjiFileMapInput, JikjiFileMapSummary } from "./file-map.ts";
export {
	JIKJI_FILE_MAP_FIELD_CHAR_CAP,
	JIKJI_FILE_MAP_ITEM_CAP,
	JIKJI_FILE_MAP_TOTAL_CHAR_CAP,
	parseJikjiFileMapStdout,
	renderJikjiFileMapContext,
	summarizeJikjiFileMaps,
	summarizeJikjiFileMapsBySource,
} from "./file-map.ts";
export type { JikjiSourceRoot } from "./path-map.ts";
export { mapJikjiPath, planJikjiSourceRoots, resolveReturnedPath } from "./path-map.ts";
export type {
	JikjiFailureReason,
	JikjiOptions,
	JikjiPrepareOptions,
	JikjiPrepareResult,
} from "./types.ts";
export { DEFAULT_JIKJI_OPTIONS } from "./types.ts";
