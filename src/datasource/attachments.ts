/**
 * Attachment folders: where a datasource's saved files (photos, documents, voice notes) live on disk.
 *
 * Every datasource skill may implement `DatasourceSkill.attachmentRoots()`. The agent reads it in two places:
 * the skill manifest (`load_datasource_skill`) and the tail of each `search_datasource_<id>` result, so it can
 * open or `bash`-inspect a file a message refers to. The datasource's own tool results stay path-opaque;
 * these folders are explicit, separate context.
 *
 * Adding a datasource: add one entry to {@link ATTACHMENT_RESOLVERS} keyed by its template name. Nothing else
 * changes: the factory wraps the built skill, and the manifest and search tool pick the roots up from there.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve, sep } from "node:path";
import { parse as parseToml } from "smol-toml";
import type { RetrievalMethod } from "../retrieval/types.ts";
import { discordAttachmentsRoot } from "./skills/discrawl/attachments.ts";
import type {
	DatasourceAttachmentRoot,
	DatasourceGrepHit,
	DatasourceGrepOptions,
	DatasourceIndexOptions,
	DatasourceIndexResult,
	DatasourceSkill,
	DatasourceSkillDescriptor,
	DatasourceSkillManifest,
	PollingMetadata,
	SearchFilterSpec,
	SourceDescription,
} from "./types.ts";

export interface AttachmentResolveInput {
	/** Built-in template name (`telegram`, `discord`, ...), not the connection alias. */
	readonly template: string;
	/** The connection's name in the config; names per-connection folders. */
	readonly registrationName: string;
	/** The connection's config entry (only `connector` is read). */
	readonly config: { readonly connector?: Record<string, unknown> };
	readonly home: string;
	readonly env: NodeJS.ProcessEnv;
}

type AttachmentResolver = (input: AttachmentResolveInput) => readonly DatasourceAttachmentRoot[];

function stringOption(connector: Record<string, unknown> | undefined, key: string): string | undefined {
	const value = connector?.[key];
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function expandHome(path: string, home: string): string {
	if (path === "~") return home;
	return path.startsWith("~/") || (sep === "\\" && path.startsWith("~\\")) ? join(home, path.slice(2)) : path;
}

/** `media/` beside a crawler's archive database: where `telecrawl`/`wacrawl` keep media. */
function mediaBesideArchive(input: AttachmentResolveInput, defaultDb: string): string {
	const configured = stringOption(input.config.connector, "databasePath");
	return join(dirname(expandHome(configured ?? defaultDb, input.home)), "media");
}

const ATTACHMENT_RESOLVERS: Readonly<Record<string, AttachmentResolver>> = {
	discord: (input) => [
		{
			id: "downloads",
			path: discordAttachmentsRoot(input.registrationName, input.env),
			description:
				"Discord attachments saved by AutoRAG as <server>/<channel>/<file>. download-status.json lists every attachment with its state.",
		},
	],
	telegram: (input) => [
		{
			id: "media",
			path: mediaBesideArchive(input, join(input.home, ".telecrawl", "telecrawl.db")),
			description:
				"Telegram media archived by telecrawl (cached locally; cloud media only when imported with --fetch-media).",
		},
	],
	whatsapp: (input) => {
		const roots: DatasourceAttachmentRoot[] = [
			{
				id: "media",
				path: mediaBesideArchive(input, join(input.home, ".wacrawl", "wacrawl.db")),
				description: "WhatsApp media copied by wacrawl (only filled when imported with --copy-media).",
			},
		];
		const source =
			stringOption(input.config.connector, "sourcePath") ??
			join(input.home, "Library", "Group Containers", "group.net.whatsapp.WhatsApp.shared");
		roots.push({
			id: "desktop-media",
			path: join(expandHome(source, input.home), "Message", "Media"),
			description: "WhatsApp Desktop's own media store, read-only; the archive's media_path points here.",
		});
		return roots;
	},
	slack: (input) => [
		{
			id: "media",
			path: join(slackCacheDir(input), "media"),
			description: "Slack files fetched by slacrawl (`slacrawl files fetch`); files not fetched yet are absent.",
		},
	],
	kakao: (input) => {
		const dataDir = stringOption(input.config.connector, "workspacePath");
		return [
			{
				id: "media",
				path: join(
					dataDir === undefined
						? join(input.home, "Library", "Application Support", "katok")
						: expandHome(dataDir, input.home),
					"media",
				),
				description:
					"KakaoTalk media saved by lazykatok, one <chatId> folder per room (`lazykatok media backfill`).",
			},
		];
	},
};

/** slacrawl's `cache_dir`: from its TOML config when readable, else `~/.slacrawl/cache`. */
function slackCacheDir(input: AttachmentResolveInput): string {
	const fallback = join(input.home, ".slacrawl", "cache");
	const configured = stringOption(input.config.connector, "configPath");
	const file =
		configured === undefined ? join(input.home, ".slacrawl", "config.toml") : expandHome(configured, input.home);
	try {
		const value = parseToml(readFileSync(file, "utf8")).cache_dir;
		if (typeof value !== "string" || value.length === 0) return fallback;
		const expanded = expandHome(value, input.home);
		return isAbsolute(expanded) ? expanded : resolve(dirname(file), expanded);
	} catch {
		return fallback;
	}
}

/** Attachment folders for one configured connection; empty for a datasource that holds no files. */
export function resolveAttachmentRoots(input: AttachmentResolveInput): readonly DatasourceAttachmentRoot[] {
	return Object.hasOwn(ATTACHMENT_RESOLVERS, input.template)
		? (ATTACHMENT_RESOLVERS[input.template]?.(input) ?? [])
		: [];
}

function folderState(path: string): "has files" | "empty" | "not created yet" {
	if (!existsSync(path)) return "not created yet";
	try {
		return readdirSync(path).length > 0 ? "has files" : "empty";
	} catch {
		return "not created yet";
	}
}

/** Bullet list of folders with their current state; empty string when there are none. */
export function formatAttachmentRoots(roots: readonly DatasourceAttachmentRoot[]): string {
	return roots.map((root) => `- ${root.path} (${folderState(root.path)}): ${root.description}`).join("\n");
}

/** Section appended to a skill manifest. */
function manifestSection(roots: readonly DatasourceAttachmentRoot[]): string {
	return [
		"## Attachments",
		"Files this datasource saved locally. A message that mentions a photo, file, or voice note may have it here; inspect it with your file tools (`read`, `bash`).",
		formatAttachmentRoots(roots),
	].join("\n");
}

/**
 * Gives a built skill its attachment folders and mentions them in its manifest. Applied outermost, after
 * aliasing and description wrapping, so alias rewriting never touches the folder paths.
 */
export class AttachmentRootsSkill implements DatasourceSkill {
	/** The wrapped skill, for callers that need its concrete type. */
	readonly inner: DatasourceSkill;
	private readonly roots: readonly DatasourceAttachmentRoot[];

	constructor(skill: DatasourceSkill, roots: readonly DatasourceAttachmentRoot[]) {
		this.inner = skill;
		this.roots = roots;
	}

	describe(): DatasourceSkillDescriptor {
		return this.inner.describe();
	}

	polling(): PollingMetadata {
		return this.inner.polling();
	}

	index(options?: DatasourceIndexOptions): Promise<DatasourceIndexResult> {
		return this.inner.index(options);
	}

	retrievalMethods(): readonly RetrievalMethod[] {
		return this.inner.retrievalMethods();
	}

	describeSources(): readonly SourceDescription[] {
		return this.inner.describeSources();
	}

	searchFilters(): Promise<readonly SearchFilterSpec[]> {
		return this.inner.searchFilters?.() ?? Promise.resolve([]);
	}

	attachmentRoots(): readonly DatasourceAttachmentRoot[] {
		return this.roots;
	}

	grep(pattern: string, options?: DatasourceGrepOptions): Promise<readonly DatasourceGrepHit[]> {
		return this.inner.grep?.(pattern, options) ?? Promise.resolve([]);
	}

	skillManifest(): DatasourceSkillManifest {
		const manifest = this.inner.skillManifest();
		return this.roots.length === 0
			? manifest
			: { ...manifest, content: `${manifest.content}\n\n${manifestSection(this.roots)}` };
	}
}
