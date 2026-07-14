import { realpath, stat } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import {
	createFindToolDefinition,
	createGrepToolDefinition,
	createLsToolDefinition,
	createReadToolDefinition,
	type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";

const PARENT_TRAVERSAL_SEGMENT = /(^|[/\\])\.\.($|[/\\])/;
const TOOL_METADATA_CWD = "/";

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
			return createReadToolDefinition(root).execute(toolCallId, { ...params, path: target }, signal, onUpdate, ctx);
		},
	});

	const grepTool = createGrepToolDefinition(TOOL_METADATA_CWD);
	pi.registerTool({
		...grepTool,
		async execute(toolCallId, params, signal, onUpdate, ctx) {
			const { root, target } = await resolveContainedPath(await getPinnedRoot(ctx.cwd), params.path ?? ".");
			return createGrepToolDefinition(root).execute(toolCallId, { ...params, path: target }, signal, onUpdate, ctx);
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
