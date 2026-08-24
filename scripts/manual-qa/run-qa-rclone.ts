/**
 * Manual QA for issue #1453.
 *
 * This is intentionally process-boundary shaped but credential-free: the
 * injected runner behaves like `rclone lsjson` and `rclone copyto`, allowing
 * the full datasource/agent surface to be exercised deterministically.
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
import { CloudDriveSkill } from "../../src/datasource/skills/cloud-drive/skill.ts";
import type { RcloneRunResult } from "../../src/datasource/skills/gdrive/rclone-connector.ts";
import { RcloneConnector } from "../../src/datasource/skills/gdrive/rclone-connector.ts";

const root = mkdtempSync(join(tmpdir(), "autorag-rclone-manual-qa-"));
let failures = 0;
const check = (name: string, pass: boolean, note = "") => {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
};

let version = 1;
let failCopy = false;
const listings = () =>
	JSON.stringify(
		version === 1
			? [
					{ Path: "reports/q3.md", Name: "q3.md", Size: 20, Hashes: { md5: "q3-v1" }, ModTime: "2026-08-01T00:00:00Z" },
					{ Path: "old.md", Name: "old.md", Size: 10, Hashes: { md5: "old-v1" }, ModTime: "2026-08-01T00:00:00Z" },
				]
			: [{ Path: "reports/q3-renamed.md", Name: "q3-renamed.md", Size: 20, Hashes: { md5: "q3-v2" }, ModTime: "2026-08-02T00:00:00Z" }],
	);

const runner = async (args: readonly string[]): Promise<RcloneRunResult> => {
	if (args[0] === "lsjson") return { ok: true, stdout: listings(), stderr: "", code: 0 };
	if (args[0] === "copyto") {
		if (failCopy) return { ok: false, stdout: "", stderr: "interrupted", code: 1 };
		writeFileSync(args[2] ?? "", `content for ${args[1]}`);
		return { ok: true, stdout: "", stderr: "", code: 0 };
	}
	return { ok: false, stdout: "", stderr: "unexpected", code: 1 };
};

try {
	const connector = new RcloneConnector({
		remote: "gdrive:",
		skillName: "cloud-drive",
		workspaceRoot: root,
		instanceId: "manual",
		runner,
	});
	const skill = new CloudDriveSkill({
		instanceId: "manual",
		workspaceRoot: root,
		connector,
	});

	const initial = await skill.index();
	check("initial sync indexes files", initial.ok && initial.chunkCount === 2);
	const noop = await skill.index();
	check("no-op sync changes nothing", noop.ok && noop.chunkCount === 2);

	version = 2;
	const update = await skill.index();
	check("rename removes old virtual file and indexes new file", update.ok && update.chunkCount === 1);
	const [method] = skill.retrievalMethods();
	const renamed = await method?.retrieve("q3-renamed", { topK: 3 });
	check("search finds renamed file", (renamed?.length ?? 0) > 0);

	version = 1;
	failCopy = true;
	const interrupted = await skill.index();
	check("interrupted sync reports failure", !interrupted.ok);
	const previous = await method?.retrieve("q3-renamed", { topK: 3 });
	check("previous snapshot remains searchable after failure", (previous?.length ?? 0) > 0);

	failCopy = false;
	const tool = createSearchDatasourceDocumentsTool({
		async searchDatasourceDocuments(query, options) {
			const results = (await method?.retrieve(query, { topK: options?.topK ?? 5, scope: options?.scope })) ?? [];
			return { results, diagnostics: [] };
		},
	});
	const toolResult = await tool.execute("manual-qa", { query: "q3-renamed", scope: "/cloud-drive/manual/**" });
	check("search_datasource_documents returns scoped cloud-drive hit", toolResult.details.resultCount > 0);
	check("manifest has stable virtual metadata", readFileSync(join(root, ".autorag", "datasources", "cloud-drive", "manual", "manifest.json"), "utf8").includes("reports"));
} finally {
	rmSync(root, { recursive: true, force: true });
}

console.log(failures === 0 ? "\nRCLONE MANUAL QA PASSED" : `\nRCLONE MANUAL QA: ${failures} failure(s)`);
if (failures > 0) process.exitCode = 1;
