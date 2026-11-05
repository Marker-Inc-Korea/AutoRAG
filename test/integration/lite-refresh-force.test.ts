import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

let root: string;
let docs: string;
let binary: string;
let log: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-lite-refresh-force-"));
	docs = join(root, "docs");
	binary = join(root, "minsync");
	log = join(root, "commands.jsonl");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "note.md"), "Force refresh fixture\n");
	writeFileSync(
		binary,
		`#!/bin/sh
printf '%s\\n' "$*" >> ${JSON.stringify(log)}
case "$1" in
  stage)
    shift
    state_dir=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --state-dir) state_dir="$2"; shift 2 ;;
        --state-dir=*) state_dir="\${1#--state-dir=}"; shift ;;
        *) shift ;;
      esac
    done
    exec ${JSON.stringify(process.execPath)} -e 'require(process.argv[1]).runStage({ stateDir: process.argv[2] });' ${JSON.stringify(fakeMinSyncStageSimulatorPath)} "$state_dir"
    ;;
  init) mkdir -p .minsync; printf '%s\\n' '[embedder]' 'id = "fixture"' > .minsync/config.toml; exit 0 ;;
  check) printf '%s\\n' '{"vectorstore_ok":true,"embedder_ok":true}'; exit 0 ;;
  sync) mkdir -p .minsync; printf '%s\\n' '{}' > .minsync/cursor.json; printf '%s\\n' '{"files_processed":1}'; exit 0 ;;
esac
exit 2
`,
	);
	chmodSync(binary, 0o755);
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("model-free full refresh", () => {
	it("passes force through to MinSync after an incremental cursor exists", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { binaryPath: binary, workspacePath: join(root, ".autorag", "minsync"), autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
			dupey: false,
			excludeExactDuplicates: false,
		});

		await agent.refresh(false);
		await agent.refresh(true);

		const syncCommands = readFileSync(log, "utf8")
			.trim()
			.split("\n")
			.filter((command) => command.startsWith("sync"));
		expect(syncCommands).toEqual([
			"sync --full --format json --progress jsonl",
			"sync --full --format json --progress jsonl",
		]);
	});
});
