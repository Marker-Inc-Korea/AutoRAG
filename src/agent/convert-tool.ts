import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { ConvertAndRelateResult, ConvertedCopyLink, FileeConvertRequest } from "../convert/index.ts";

export const CONVERT_FILES_TOOL_NAME = "convert_files";
export const LIST_FILE_CONVERSIONS_TOOL_NAME = "list_file_conversions";

/** Caps keep a huge batch from flooding the model's context. */
const MAX_RENDERED_FILES = 200;

export type ConvertFilesDetails = ConvertAndRelateResult;

export interface FileConversionProvider {
	convertFiles(request: FileeConvertRequest, signal?: AbortSignal): Promise<ConvertFilesDetails>;
	listFileConversions(from?: string, signal?: AbortSignal): Promise<string>;
}

const convertFilesSchema = Type.Object({
	inputs: Type.Array(Type.String(), {
		minItems: 1,
		description: "Files, folders or wildcards (*.heic) to convert. Any path on this machine.",
	}),
	to: Type.String({
		description:
			"Target format id or extension: pdf, docx, hwpx, md, html, txt, epub, png, jpg, webp, mp4, mp3, ... Use list_file_conversions when unsure what a source can become.",
	}),
	outputDir: Type.Optional(Type.String({ description: "Write results here instead of next to each source." })),
	recursive: Type.Optional(Type.Boolean({ description: "Include sub folders of folder inputs." })),
	quality: Type.Optional(
		Type.Integer({ minimum: 1, maximum: 100, description: "Quality for lossy image formats (jpg, webp, avif)." }),
	),
	overwrite: Type.Optional(
		Type.Boolean({
			description:
				"Replace an existing output file. Off by default: an existing name gets a number added instead, so nothing is lost.",
		}),
	),
	skipExisting: Type.Optional(
		Type.Boolean({ description: "Leave an existing output file alone and skip that input." }),
	),
	keepDates: Type.Optional(Type.Boolean({ description: "Give each output the modified time of its source." })),
});

export function createConvertFilesTool(
	provider: FileConversionProvider,
): AgentTool<typeof convertFilesSchema, ConvertFilesDetails> {
	return {
		name: CONVERT_FILES_TOOL_NAME,
		label: "Convert Files",
		description:
			"Convert local files to another format offline with the Filee command line (documents, spreadsheets, slides, HWP/HWPX, images, e-books, archives, audio, video, data, subtitles; 240+ formats). Works on any path. Reports each output path and whether dupey sees the converted copy as the same document as its source.",
		parameters: convertFilesSchema,
		async execute(_id, params, signal): Promise<AgentToolResult<ConvertFilesDetails>> {
			const details = await provider.convertFiles(params, signal);
			const result: AgentToolResult<ConvertFilesDetails> = {
				content: [{ type: "text", text: renderConvertReport(details) }],
				details,
			};
			if (details.result.converted === 0 && details.result.failed > 0) result.isError = true;
			return result;
		},
	};
}

const listConversionsSchema = Type.Object({
	from: Type.Optional(
		Type.String({
			description: "Source format id or extension (heic, docx, hwp, mkv). Omit to list every known format.",
		}),
	),
});

export function createListFileConversionsTool(
	provider: FileConversionProvider,
): AgentTool<typeof listConversionsSchema, { readonly from?: string }> {
	return {
		name: LIST_FILE_CONVERSIONS_TOOL_NAME,
		label: "List File Conversions",
		description:
			"List the file formats the converter knows, or what one source format can be converted to (including targets that need an optional engine). Read-only.",
		parameters: listConversionsSchema,
		async execute(_id, params, signal) {
			const text = await provider.listFileConversions(params.from, signal);
			return {
				content: [{ type: "text", text }],
				details: params.from === undefined ? {} : { from: params.from },
			};
		},
	};
}

/**
 * Model-facing report. `details` is UI/log-only in pi-agent-core, so every
 * output path, failure and dupey verdict is rendered into the text the model reads.
 */
function renderConvertReport(details: ConvertFilesDetails): string {
	const { result, links } = details;
	const lines = [
		`Filee converted to ${result.target}: converted=${result.converted} skipped=${result.skipped} failed=${result.failed}`,
	];
	const byOutput = new Map<string, ConvertedCopyLink>();
	for (const link of links) byOutput.set(link.output, link);
	let rendered = 0;
	for (const file of result.files) {
		if (rendered >= MAX_RENDERED_FILES) break;
		rendered += 1;
		if (file.state === "done") {
			for (const output of file.outputs) {
				const link = byOutput.get(output);
				const verdict = link === undefined ? "" : ` [${describeRelation(link)}]`;
				lines.push(`  ok    ${file.source} -> ${output}${verdict}${file.notice ? ` (${file.notice})` : ""}`);
			}
		} else if (file.state === "skipped") {
			lines.push(`  skip  ${file.source} (output already exists)`);
		} else {
			lines.push(`  fail  ${file.source}: ${file.error ?? "conversion failed"}`);
		}
	}
	if (result.files.length > rendered) lines.push(`  … ${result.files.length - rendered} more file(s) omitted`);
	return lines.join("\n");
}

function describeRelation(link: ConvertedCopyLink): string {
	const relation = link.dupey;
	switch (relation.relation) {
		case "exact":
			return "dupey: exact — same canonical text as the source; an exact duplicate family";
		case "near":
			return `dupey: near (jaccard=${relation.jaccard}) — same document in another format; near family, review before treating as duplicate`;
		case "contains":
			return `dupey: contains (jaccard=${relation.jaccard}) — one holds most of the other`;
		case "different":
			return `dupey: different (near=${relation.nearScore}) — not grouped with the source; extracted text differs (layout-only formats such as PDF often do)`;
		case "unsupported":
			return "dupey: unsupported — dupey cannot read this format, so it will never be grouped with the source";
		case "unavailable":
			return `dupey: unavailable (${relation.reason})`;
	}
}
