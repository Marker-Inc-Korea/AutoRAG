import { type RuntimeStatus, runtimeStatus, stopRuntime } from "../../embedding-runtime/index.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export interface GatewayCommandDeps {
	runtimeStatus?: () => Promise<RuntimeStatus>;
	stopRuntime?: () => Promise<void>;
}
export async function runGateway(ctx: CommandContext, deps: GatewayCommandDeps = {}): Promise<number> {
	const sub = ctx.positionals[0];
	const json = ctx.json || ctx.flags.format === "json";
	try {
		if (sub === "status") {
			const status = await (deps.runtimeStatus ?? runtimeStatus)();
			ctx.stdout(json ? JSON.stringify(status, null, 2) : renderStatus(status));
			return 0;
		}
		if (sub === "stop") {
			await (deps.stopRuntime ?? stopRuntime)();
			ctx.stdout(json ? JSON.stringify({ ok: true, state: "stopped" }, null, 2) : "gateway: stopped");
			return 0;
		}
		throw new UsageError("Usage: autorag gateway <status|stop> [--format json]");
	} catch (error) {
		ctx.stderr(renderError(error, { json, debug: ctx.debug }));
		return error instanceof UsageError ? 2 : 1;
	}
}
function renderStatus(status: RuntimeStatus): string {
	const lines = [`gateway: ${status.state}`, `  backend: ${status.backend}`, `  model: ${status.model}`];
	if (status.profileId) lines.push(`  profile: ${status.profileId}`);
	lines.push(`  health: ${status.health.ok ? "ok" : status.health.code}`);
	return lines.join("\n");
}
class UsageError extends Error {}
