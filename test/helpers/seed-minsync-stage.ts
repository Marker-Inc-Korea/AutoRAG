import { copyFileSync, mkdirSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { lookupMinSyncInPath } from "../../src/minsync/binary.ts";
import { MinSyncStageSession } from "../../src/minsync/stage.ts";
import { mirrorStageKey, type ParsedMirrorIndex } from "../../src/mirror/index-store.ts";
import { parsedMirrorRoot, parsedOutputPath } from "../../src/mirror/paths.ts";

/** Seed through the engine contract, not by manufacturing its private state files. */
export async function seedMinSyncStage(root: string, index: ParsedMirrorIndex): Promise<void> {
	const binary = lookupMinSyncInPath(process.env);
	if (binary === undefined) throw new Error("test MinSync executable is missing");
	const entries = Object.values(index.entries);
	const byKey = new Map(entries.map((entry) => [mirrorStageKey(entry.virtualPath), entry]));
	const items = entries.map((entry) => {
		const { outputPath: _output, updatedAt: _updated, parserFingerprint: _fingerprint, ...source } = entry;
		return { key: mirrorStageKey(entry.virtualPath), version: JSON.stringify(source) };
	});
	const session = new MinSyncStageSession(binary, join(parsedMirrorRoot(root), "stage"), root);
	try {
		const plan = await session.open({
			stage: "parse",
			source_id: "autorag-parsed-mirror",
			fingerprint: "fixture",
			output_root: join(parsedMirrorRoot(root), "files"),
			items,
		});
		for (const decision of plan.decisions) {
			if (decision.action === "delete") {
				await session.delete(decision.key);
				continue;
			}
			if (decision.action !== "process" && decision.action !== "retry") continue;
			const entry = byKey.get(decision.key)!;
			const outputPath = parsedOutputPath(root, entry.virtualPath);
			mkdirSync(dirname(outputPath), { recursive: true });
			if (outputPath !== entry.outputPath) copyFileSync(entry.outputPath, outputPath);
			await session.commit(decision.key, decision.version!, { path: basename(outputPath) });
		}
		await session.finish();
	} finally {
		await session.close();
	}
}
