import { spawn } from "node:child_process";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { findAgent } from "./agents.ts";

export interface OrganizeResult {
	status: "delegated" | "unavailable";
	agent?: string;
	reason?: string;
}

/** A project root string or a provider resolved at call time. */
export type CwdProvider = string | (() => string);

function resolveCwd(provider: CwdProvider): string {
	return typeof provider === "function" ? provider() : provider;
}

/** Child-`pi` delegation is opt-in to keep CI/sandboxes hermetic. */
function spawnEnabled(): boolean {
	return process.env.AUTORAG_E2E_SPAWN === "1";
}

function delegateViaChildPi(cwd: string, agent: string, task: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const child = spawn("pi", ["--agent", agent, "--print", task], { cwd, stdio: "ignore" });
		child.on("error", reject);
		child.on("exit", (code) => {
			if (code === 0) resolve();
			else reject(new Error(`pi exited with code ${code}`));
		});
	});
}

/**
 * Delegate a reorganization task to the organizer sub-agent via a child `pi`
 * process. Always resolves to a structured result — never throws — so a missing
 * agent definition or an unavailable `pi` binary degrades gracefully.
 *
 * NOTE: the concrete organizing pipeline is a skeleton; this is the delegation
 * entry point only.
 */
export async function runOrganizer(cwd: string, task: string): Promise<OrganizeResult> {
	const organizer = findAgent(cwd, "organizer");
	if (!organizer) {
		return { status: "unavailable", reason: "organizer agent definition not found" };
	}
	if (!spawnEnabled()) {
		return { status: "unavailable", reason: "child-pi delegation disabled; set AUTORAG_E2E_SPAWN=1 to enable" };
	}
	try {
		await delegateViaChildPi(cwd, organizer.name, task);
		return { status: "delegated", agent: organizer.name };
	} catch (error) {
		return { status: "unavailable", reason: error instanceof Error ? error.message : String(error) };
	}
}

const organizeSchema = Type.Object({
	task: Type.String({ description: "What to reorganize in the virtual document layout" }),
});

function resultText(result: OrganizeResult): string {
	return result.status === "delegated"
		? `Delegated organize task to sub-agent '${result.agent}'.`
		: `Organizer unavailable: ${result.reason}`;
}

export function createOrganizeToolDefinition(
	cwdProvider: CwdProvider,
): ToolDefinition<typeof organizeSchema, OrganizeResult> {
	return {
		name: "organize",
		label: "Organize (sub-agent)",
		description:
			"Delegate document-layout reorganization to the organizer sub-agent (uses agentdir virtual ops; source files are never modified).",
		promptSnippet: "Delegate reorganizing the virtual document layout to the organizer sub-agent",
		promptGuidelines: [
			"Use organize to restructure the virtual tree into a clearer layout. Source files are never modified.",
		],
		parameters: organizeSchema,
		async execute(_toolCallId, params) {
			const result = await runOrganizer(resolveCwd(cwdProvider), params.task);
			return { content: [{ type: "text", text: resultText(result) }], details: result };
		},
	};
}

export function createOrganizeTool(cwdProvider: CwdProvider): AgentTool<typeof organizeSchema, OrganizeResult> {
	const def = createOrganizeToolDefinition(cwdProvider);
	return {
		name: def.name,
		label: def.label,
		description: def.description,
		parameters: def.parameters,
		async execute(toolCallId, params, signal, onUpdate) {
			return def.execute(toolCallId, params, signal, onUpdate, undefined as never);
		},
	};
}
