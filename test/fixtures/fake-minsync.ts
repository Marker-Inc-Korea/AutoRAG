import { chmodSync, copyFileSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const FAKE_MINSYNC_SCRIPT = fileURLToPath(new URL("./fake-minsync.mjs", import.meta.url));

export interface FakeMinSync {
	readonly binaryPath: string;
	/** Make every `minsync sync` exit 1 with "boom" until {@link FakeMinSync.healSync}. */
	readonly breakSync: () => void;
	readonly healSync: () => void;
	/** Make `minsync query --mode hybrid` fail (a down embedder) while vector and bm25 queries keep working. */
	readonly breakHybrid: () => void;
}

/** Copy the fake `minsync` CLI into `directory`; its call log and failure flags live beside it. */
export function installFakeMinSync(directory: string): FakeMinSync {
	mkdirSync(directory, { recursive: true });
	const binaryPath = join(directory, "fake-minsync.mjs");
	copyFileSync(FAKE_MINSYNC_SCRIPT, binaryPath);
	chmodSync(binaryPath, 0o755);
	const flag = (name: string) => join(directory, name);
	return {
		binaryPath,
		breakSync: () => writeFileSync(flag("fail-sync"), ""),
		healSync: () => rmSync(flag("fail-sync"), { force: true }),
		breakHybrid: () => writeFileSync(flag("fail-hybrid"), ""),
	};
}
