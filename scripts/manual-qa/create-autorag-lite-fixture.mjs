#!/usr/bin/env node
/**
 * create-autorag-lite-fixture.mjs
 *
 * Repository-owned deterministic fixture generator for AutoRAG-lite QA.
 *
 * Creates a hermetic fixture tree under `--root` (default: /tmp/autorag-lite-fixture)
 * with:
 *   - docs/README.md           Sample local documents (the indexable content)
 *   - config.json              Trusted AutoRAG config (workspace, search paths,
 *                              a datasource with a deliberately missing binary)
 *   - unrefreshed-config.json  Same shape but with different description
 *                              (simulates a fresh, never-refreshed project)
 *   - workspace/               Isolated workspace directory
 *   - workspace/data/          Small data files for indexing
 *   - workspace/src/           Sample source code
 *
 * The fixture is deterministic, non-interactive, requires no network,
 * MinSync, Jikji, or Ollama, and contains no secrets.
 *
 * Usage:
 *   node scripts/manual-qa/create-autorag-lite-fixture.mjs
 *   node scripts/manual-qa/create-autorag-lite-fixture.mjs --root /tmp/my-fixture
 *   node scripts/manual-qa/create-autorag-lite-fixture.mjs --root /tmp/my-fixture --print-json
 *   node scripts/manual-qa/create-autorag-lite-fixture.mjs --root /tmp/my-fixture --cleanup
 */

import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// ── CLI argument parsing ────────────────────────────────────────────────
const args = process.argv.slice(2);

let root = "/tmp/autorag-lite-fixture";
let printJson = false;
let cleanup = false;

for (let i = 0; i < args.length; i++) {
	switch (args[i]) {
		case "--root": {
			i++;
			if (i >= args.length) {
				error("--root requires a path argument");
				process.exit(2);
			}
			root = args[i];
			if (root.length === 0) {
				error("--root path must not be empty");
				process.exit(2);
			}
			if (root.startsWith("-")) {
				error(`--root path looks like a flag: ${root}`);
				process.exit(2);
			}
			break;
		}
		case "--print-json":
			printJson = true;
			break;
		case "--cleanup":
			cleanup = true;
			break;
		default:
			error(`unknown flag: ${args[i]}`);
			process.exit(2);
	}
}

// ── Deterministic file contents ─────────────────────────────────────────
const DOCS_README = `# AutoRAG-lite QA Fixture

This directory contains deterministic test documents for verifying
AutoRAG-lite retrieval and indexing behavior.

## Project Context

The AutoRAG-lite module is a loop-free facade that reuses existing
refresh, retrieval, datasource, MinSync, Jikji, evidence, and feedback
contracts without requiring a model or agent loop.

## Notes

- All paths in config.json are relative to this fixture root.
- The "missing-cli" datasource entry deliberately sets binaryPath to
  a non-existent executable to test diagnostics.
- This fixture requires no network access, MinSync server, Jikji index,
  or Ollama instance.
`;

const NOTES_TXT = `project-todo.txt
- Refactor the retrieval merger to handle empty results
- Add integration tests for the scope filter
- Document the missing-binary diagnostic code

meeting-notes-2026-09-01.txt
- Discussed the new loop-free facade design
- Agreed to keep existing model-backed commands unchanged
- Assigned fixture creation to the automation pipeline

architecture-decisions.txt
- ADR-0001: Use Pydantic v2 for all config boundary parsing
- ADR-0002: MinSync is the sole vector store backend
- ADR-0003: All datasource skills use the external-CLI boundary pattern
`;

const SOURCE_CODE = `// main.ts
export function greet(name: string): string {
  return \`Hello, \${name}!\`;
}

export function add(a: number, b: number): number {
  return a + b;
}
`;

/** Build the trusted config section. */
function buildConfig(baseRoot) {
	return {
		searchPaths: [join(baseRoot, "docs"), join(baseRoot, "workspace", "data")],
		workspacePath: join(baseRoot, "workspace"),
		memoryPath: join(baseRoot, "workspace", "memory.db"),
		datasources: {
			"missing-cli": {
				type: "obsidian",
				description:
					"A deliberately misconfigured datasource whose external CLI binary does not exist on disk. Used to test graceful diagnostics without affecting other refresh methods.",
				connector: {
					vaultPath: join(baseRoot, "workspace"),
					binaryPath: "/definitely/missing/autorag-lite-cli",
					configPath: "/definitely/missing/config",
				},
				tags: ["test-fixture", "missing-binary"],
			},
		},
		minSync: {
			autoInstall: false,
		},
		jikji: false,
	};
}

// ── Run ─────────────────────────────────────────────────────────────────

function error(msg) {
	const full = `error: ${msg}`;
	if (printJson) {
		process.stdout.write(JSON.stringify({ ok: false, error: msg }));
	} else {
		console.error(full);
	}
}

// Cleanup mode
if (cleanup) {
	if (!existsSync(root)) {
		if (printJson) {
			process.stdout.write(JSON.stringify({ ok: true, cleaned: false, reason: "does not exist", root }));
		} else {
			console.log(`no-op: fixture root does not exist: ${root}`);
		}
		process.exit(0);
	}
	rmSync(root, { recursive: true, force: true });
	if (printJson) {
		process.stdout.write(JSON.stringify({ ok: true, cleaned: true, root }));
	} else {
		console.log(`cleaned: ${root}`);
	}
	process.exit(0);
}

// Reject pre-existing state
if (existsSync(root)) {
	error(`fixture root already exists: ${root}`);
	process.exit(1);
}

// Build fixture tree
mkdirSync(join(root, "docs"), { recursive: true });
mkdirSync(join(root, "workspace", "data"), { recursive: true });
mkdirSync(join(root, "workspace", "src"), { recursive: true });

writeFileSync(join(root, "docs", "README.md"), DOCS_README);
writeFileSync(join(root, "workspace", "data", "notes.txt"), NOTES_TXT);
writeFileSync(join(root, "workspace", "src", "main.ts"), SOURCE_CODE);

// Same config with different description for "unrefreshed" variant
const config = buildConfig(root);
const configJson = JSON.stringify(config, null, 2) + "\n";
writeFileSync(join(root, "config.json"), configJson);

const unrefreshedConfig = {
	searchPaths: [join(root, "docs"), join(root, "workspace", "data")],
	workspacePath: join(root, "workspace"),
	memoryPath: join(root, "workspace", "memory.db"),
	datasources: {
		"missing-cli": {
			type: "obsidian",
			description:
				"A deliberately misconfigured datasource for un-refreshed fixtures. Environment has never been initialized.",
			connector: {
				vaultPath: join(root, "workspace"),
				binaryPath: "/definitely/missing/autorag-lite-cli",
				configPath: "/definitely/missing/config",
			},
			tags: ["test-fixture", "missing-binary"],
		},
	},
	minSync: {
		autoInstall: false,
	},
	jikji: false,
};

const unrefreshedJson = JSON.stringify(unrefreshedConfig, null, 2) + "\n";
writeFileSync(join(root, "unrefreshed-config.json"), unrefreshedJson);

// ── Output ──────────────────────────────────────────────────────────────
const output = {
	ok: true,
	root,
	contents: {
		"config.json": { keys: Object.keys(config) },
		"unrefreshed-config.json": { keys: Object.keys(unrefreshedConfig) },
		"docs/README.md": { size: DOCS_README.length },
		"workspace/data/notes.txt": { size: NOTES_TXT.length },
		"workspace/src/main.ts": { size: SOURCE_CODE.length },
	},
	datasources: Object.keys(config.datasources),
	missingBinaryPath: config.datasources["missing-cli"].connector.binaryPath,
	cleanupCommand: `node scripts/manual-qa/create-autorag-lite-fixture.mjs --root ${root} --cleanup`,
};

if (printJson) {
	process.stdout.write(JSON.stringify(output) + "\n");
} else {
	for (const line of [
		`fixture created at: ${root}`,
		`  docs/README.md              ${DOCS_README.length} bytes`,
		`  workspace/data/notes.txt    ${NOTES_TXT.length} bytes`,
		`  workspace/src/main.ts       ${SOURCE_CODE.length} bytes`,
		`  config.json                 ${configJson.length} bytes (${Object.keys(config).length} keys)`,
		`  unrefreshed-config.json     ${unrefreshedJson.length} bytes`,
		`datasource: missing-cli`,
		`  binaryPath: /definitely/missing/autorag-lite-cli (deliberately missing)`,
		`cleanup: node ${process.argv[1]} --root ${root} --cleanup`,
	]) {
		console.log(line);
	}
}