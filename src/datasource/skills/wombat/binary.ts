import { accessSync, constants } from "node:fs";
import { delimiter, join } from "node:path";
import { resolveUserHome } from "../../../config/home.ts";

const EXECUTABLE = process.platform === "win32" ? "wombat.exe" : "wombat";

function executable(path: string): boolean {
	try {
		accessSync(path, constants.X_OK);
		return true;
	} catch {
		return false;
	}
}

/**
 * Where `cargo install wombat-sessions` puts the binary when `PATH` does not say: a desktop app or a launchd job
 * starts with a minimal `PATH` that omits `~/.cargo/bin`, so the CLI would look missing although it is installed.
 */
export function findWombatBinary(env: NodeJS.ProcessEnv = process.env): string | undefined {
	for (const dir of (env.PATH ?? "").split(delimiter)) {
		if (dir.length === 0) continue;
		const candidate = join(dir, EXECUTABLE);
		if (executable(candidate)) return candidate;
	}
	const cargoHome = env.CARGO_HOME !== undefined && env.CARGO_HOME.length > 0 ? env.CARGO_HOME : undefined;
	const candidate = join(cargoHome ?? join(resolveUserHome(env), ".cargo"), "bin", EXECUTABLE);
	return executable(candidate) ? candidate : undefined;
}
