import { spawn } from "node:child_process";
import { mkdtemp, realpath, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import {
	createFindToolDefinition,
	createGrepToolDefinition,
	createLsToolDefinition,
	createReadToolDefinition,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import {
	classifyFilesystemRoot,
	isDatalessPlaceholder,
	listMaterializedFiles,
} from "../filesystem/cloud-placeholder.ts";

const PARENT_TRAVERSAL_SEGMENT = /(^|[/\\])\.\.($|[/\\])/;
const TOOL_METADATA_CWD = "/";
const GREP_CLOUD_TIMEOUT_MS = 20_000;

type ExplorerToolRegistrar = Pick<ExtensionAPI, "registerTool">;

class ExplorerPathContainmentError extends Error {
	readonly code: "EXPLORER_PATH_CONTAINMENT";
	readonly requestedPath: string;
	readonly root: string;

	constructor(message: string, requestedPath: string, root: string) {
		super(message);
		this.name = "ExplorerPathContainmentError";
		this.code = "EXPLORER_PATH_CONTAINMENT";
		this.requestedPath = requestedPath;
		this.root = root;
	}
}

function isMissingPathError(error: unknown): boolean {
	return error instanceof Error && "code" in error && (error.code === "ENOENT" || error.code === "ENOTDIR");
}

function isEqualOrDescendant(root: string, target: string): boolean {
	const pathFromRoot = relative(root, target);
	return (
		pathFromRoot === "" ||
		(pathFromRoot !== ".." && !pathFromRoot.startsWith(`..${sep}`) && !isAbsolute(pathFromRoot))
	);
}

async function resolveExistingTargetOrNearestParent(target: string): Promise<string> {
	try {
		return await realpath(target);
	} catch (error) {
		if (!isMissingPathError(error)) throw error;
	}

	let candidateParent = dirname(target);
	while (true) {
		try {
			const realParent = await realpath(candidateParent);
			return resolve(realParent, relative(candidateParent, target));
		} catch (error) {
			if (!isMissingPathError(error)) throw error;
			const nextParent = dirname(candidateParent);
			if (nextParent === candidateParent) throw error;
			candidateParent = nextParent;
		}
	}
}

async function resolveExplorerRoot(cwd: string): Promise<string> {
	const root = await realpath(cwd);
	if (!(await stat(root)).isDirectory()) {
		throw new Error(`Explorer assigned cwd is not a directory: ${cwd}`);
	}
	return root;
}

async function resolveContainedPath(root: string, requestedPath: string): Promise<{ root: string; target: string }> {
	if (PARENT_TRAVERSAL_SEGMENT.test(requestedPath)) {
		throw new ExplorerPathContainmentError(
			`Explorer filesystem containment rejected parent traversal path: ${requestedPath}`,
			requestedPath,
			root,
		);
	}

	const target = await resolveExistingTargetOrNearestParent(resolve(root, requestedPath));
	if (!isEqualOrDescendant(root, target)) {
		throw new ExplorerPathContainmentError(
			`Explorer filesystem containment rejected path outside assigned cwd: ${requestedPath}`,
			requestedPath,
			root,
		);
	}
	return { root, target };
}

export default function registerExplorerTools(pi: ExplorerToolRegistrar): void {
	let pinnedRoot: Promise<string> | undefined;
	const getPinnedRoot = (cwd: string): Promise<string> => {
		pinnedRoot ??= resolveExplorerRoot(cwd);
		return pinnedRoot;
	};

	const readTool = createReadToolDefinition(TOOL_METADATA_CWD);
	pi.registerTool({
		...readTool,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const { root, target } = await resolveContainedPath(await getPinnedRoot(ctx.cwd), params.path);
			if (await isDatalessPlaceholder(target)) {
				return {
					content: [
						{
							type: "text" as const,
							text: `Skipped iCloud/File Provider placeholder (not hydrated): ${target}`,
						},
					],
					details: undefined,
				};
			}
			return createReadToolDefinition(root).execute(toolCallId, { ...params, path: target }, signal, onUpdate, ctx);
		},
	});

	const grepTool = createGrepToolDefinition(TOOL_METADATA_CWD);
	pi.registerTool({
		...grepTool,
		description: `${grepTool.description} Skips iCloud/OneDrive/Google Drive placeholders (UF_DATALESS) so search does not hydrate remote files.`,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const { root, target } = await resolveContainedPath(await getPinnedRoot(ctx.cwd), params.path ?? ".");
			const timeout = AbortSignal.timeout(GREP_CLOUD_TIMEOUT_MS);
			const combined = signal === undefined ? timeout : AbortSignal.any([signal, timeout]);
			const classification = await classifyFilesystemRoot(target);
			if (classification.kind === "file-provider") {
				return grepMaterializedOnly(target, params, combined);
			}
			return createGrepToolDefinition(root).execute(
				toolCallId,
				{ ...params, path: target },
				combined,
				onUpdate,
				ctx,
			);
		},
	});

	const findTool = createFindToolDefinition(TOOL_METADATA_CWD);
	pi.registerTool({
		...findTool,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const { root, target } = await resolveContainedPath(await getPinnedRoot(ctx.cwd), params.path ?? ".");
			return createFindToolDefinition(root).execute(toolCallId, { ...params, path: target }, signal, onUpdate, ctx);
		},
	});

	const lsTool = createLsToolDefinition(TOOL_METADATA_CWD);
	pi.registerTool({
		...lsTool,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const { root, target } = await resolveContainedPath(await getPinnedRoot(ctx.cwd), params.path ?? ".");
			return createLsToolDefinition(root).execute(toolCallId, { ...params, path: target }, signal, onUpdate, ctx);
		},
	});
}

async function grepMaterializedOnly(
	target: string,
	params: { pattern: string; glob?: string; ignoreCase?: boolean; literal?: boolean; limit?: number },
	signal: AbortSignal,
): Promise<{ content: Array<{ type: "text"; text: string }>; details: undefined }> {
	if (await isDatalessPlaceholder(target)) {
		return {
			content: [
				{
					type: "text",
					text: `Skipped iCloud/File Provider placeholder (not hydrated): ${target}`,
				},
			],
			details: undefined,
		};
	}
	const walk = await listMaterializedFiles(target, { timeoutMs: GREP_CLOUD_TIMEOUT_MS });
	if (walk.materialized.length === 0) {
		return {
			content: [
				{
					type: "text",
					text: `No local (materialized) files under ${target}. Skipped ${walk.skippedDataless} iCloud/File Provider placeholders so grep would not download them.`,
				},
			],
			details: undefined,
		};
	}
	const listDir = await mkdtemp(join(tmpdir(), "autorag-grep-"));
	const listPath = join(listDir, "files.txt");
	await writeFile(listPath, `${walk.materialized.join("\n")}\n`, "utf8");
	const args = ["--line-number", "--hidden", "--files-from", listPath];
	if (params.ignoreCase === true) args.push("-i");
	if (params.literal === true) args.push("-F");
	if (params.glob !== undefined && params.glob.length > 0) args.push("--glob", params.glob);
	const limit = params.limit ?? 100;
	args.push(params.pattern);
	const stdout = await runRipgrep(args, signal);
	const lines = stdout
		.split("\n")
		.filter((line) => line.length > 0)
		.slice(0, limit);
	const header = `Searched ${walk.materialized.length} local files; skipped ${walk.skippedDataless} iCloud/File Provider placeholders (not opened).`;
	return {
		content: [{ type: "text", text: `${header}\n${lines.join("\n")}` }],
		details: undefined,
	};
}

function runRipgrep(args: readonly string[], signal: AbortSignal): Promise<string> {
	return new Promise((resolvePromise, reject) => {
		const child = spawn("rg", [...args], { stdio: ["ignore", "pipe", "pipe"], signal });
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (chunk: Buffer) => {
			stdout += chunk.toString("utf8");
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString("utf8");
		});
		child.on("error", reject);
		child.on("close", (code) => {
			if (code === 0 || code === 1) {
				resolvePromise(stdout);
				return;
			}
			reject(new Error(stderr.trim() || `rg exited ${String(code)}`));
		});
	});
}
