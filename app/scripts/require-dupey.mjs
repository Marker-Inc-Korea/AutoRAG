/**
 * Hard requirement check: the AutoRAG Finder desktop app cannot provide
 * version stacks without the dupey CLI. Runs as `predev`/`prebuild` so a
 * missing dupey fails the run loudly instead of degrading the UI.
 *
 * Set AUTORAG_SKIP_DUPEY_CHECK=1 only for environments that intentionally
 * run without dupey (the app then shows a visible error banner at runtime).
 */

import { spawnSync } from "node:child_process";

const INSTALL_COMMAND = "cargo install dupey --locked";

if (process.env.AUTORAG_SKIP_DUPEY_CHECK === "1") {
	console.warn("[require-dupey] skipped: AUTORAG_SKIP_DUPEY_CHECK=1 (version stacks will show an error banner).");
	process.exit(0);
}

const probe = spawnSync("dupey", ["--version"], { encoding: "utf8" });
if (probe.status === 0) {
	process.stdout.write(`[require-dupey] ${probe.stdout.trim()}\n`);
	process.exit(0);
}

const detail = (probe.stderr || probe.error?.message || "dupey not found on PATH").trim();
process.stderr.write(
	[
		"[require-dupey] The dupey CLI is required by the AutoRAG Finder desktop app.",
		`  Reason: ${detail}`,
		`  Install: ${INSTALL_COMMAND}`,
		"  (Rust toolchain required: https://rustup.rs)",
		"",
	].join("\n"),
);
process.exit(1);
