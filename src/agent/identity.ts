import { readFileSync } from "node:fs";
import { join } from "node:path";
import { resolveAutoRAGHome } from "../config/home.ts";

/** Read a construction-time snapshot without creating or changing the user's file. */
export function readIdentity(): string | undefined {
	let text: string;
	try {
		text = readFileSync(join(resolveAutoRAGHome(), "IDENTITY.md"), "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
		throw error;
	}
	return text.trim().length > 0 ? text : undefined;
}
