import { basename, extname } from "node:path";
import { KORDOC_IMAGE_EXTENSIONS } from "./kordoc.ts";
import { type ParseInput, type ParseOutput, Parser } from "./types.ts";

/** Standalone raster image formats indexed as files alongside documents. */
export const STANDALONE_IMAGE_EXTENSIONS = KORDOC_IMAGE_EXTENSIONS;

/**
 * Lightweight parser for standalone raster images (`.png`, `.jpg`, `.jpeg`, `.webp`).
 *
 * Used when standalone image indexing is enabled without eager Stage-1 OCR (e.g., multimodal
 * embeddings like `native-multimodal:google/embeddinggemma-2`, or Stage-2 opt-in visual parsing
 * with VLM/OCR routing). Emits a minimal metadata markdown mirror naming the image so the file
 * enters the parsed-mirror index immediately in Stage 1 without blocking on OCR or VLM calls:
 *
 * - Under a multimodal embedder, `syncMinSyncWorkspace` also stages the original image file next
 *   to the mirror so MinSync embeds the pixels directly into the shared dense vector space.
 * - Under Stage-2 `visualParse` (`--method visual,minsync`), the vision decision gate inspects the
 *   image and routes screenshots/text images to OCR and visually-rich images to VLM description,
 *   materializing the supplementary `.autorag-visual.md` mirror indexed by dense + SPLADE.
 */
export class ImageMetadataParser extends Parser {
	readonly name = "image-metadata";
	readonly version = "1";
	readonly extensions: readonly string[] = STANDALONE_IMAGE_EXTENSIONS;

	async parse(input: ParseInput): Promise<ParseOutput> {
		const name = basename(input.virtualPath);
		const ext = extname(input.virtualPath).toLowerCase().replace(/^\./u, "");
		const dims = readImageDimensions(input.bytes);
		const metaParts = [
			ext.length > 0 ? `Format: ${ext.toUpperCase()}` : undefined,
			dims !== undefined ? `Dimensions: ${dims.width}x${dims.height}` : undefined,
			`Size: ${input.bytes.byteLength} bytes`,
		].filter((part): part is string => part !== undefined);

		return {
			markdown: `# ${name}\n\nImage file (${metaParts.join(", ")}).`,
			metadata: {
				parser: this.name,
				format: ext || "image",
				...(dims !== undefined ? { width: dims.width, height: dims.height } : {}),
			},
		};
	}
}

/** True when `virtualPath` names one of the supported standalone raster image formats. */
export function isStandaloneImagePath(virtualPath: string): boolean {
	const ext = extname(virtualPath).toLowerCase();
	return (STANDALONE_IMAGE_EXTENSIONS as readonly string[]).includes(ext);
}

/** Best-effort header-only dimension probe (PNG / JPEG / WebP) with zero native dependencies. */
export function readImageDimensions(
	bytes: Uint8Array,
): { readonly width: number; readonly height: number } | undefined {
	if (bytes.byteLength < 16) return undefined;
	// PNG: 8-byte signature + IHDR chunk (width @ 16..19, height @ 20..23 big-endian)
	if (bytes.byteLength >= 24 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
		const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
		const width = view.getUint32(16, false);
		const height = view.getUint32(20, false);
		if (width > 0 && height > 0) return { width, height };
		return undefined;
	}
	// WebP: RIFF....WEBP + VP8X / VP8L / VP8
	if (
		bytes.byteLength >= 30 &&
		bytes[0] === 0x52 &&
		bytes[1] === 0x49 &&
		bytes[2] === 0x46 &&
		bytes[3] === 0x46 &&
		bytes[8] === 0x57 &&
		bytes[9] === 0x45 &&
		bytes[10] === 0x42 &&
		bytes[11] === 0x50
	) {
		const tag = String.fromCharCode(bytes[12]!, bytes[13]!, bytes[14]!, bytes[15]!);
		if (tag === "VP8X") {
			const width = 1 + (bytes[24]! | (bytes[25]! << 8) | (bytes[26]! << 16));
			const height = 1 + (bytes[27]! | (bytes[28]! << 8) | (bytes[29]! << 16));
			if (width > 0 && height > 0) return { width, height };
		} else if (tag === "VP8L" && bytes.byteLength >= 25) {
			const b0 = bytes[21]!;
			const b1 = bytes[22]!;
			const b2 = bytes[23]!;
			const b3 = bytes[24]!;
			const bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
			const width = (bits & 0x3fff) + 1;
			const height = ((bits >>> 14) & 0x3fff) + 1;
			if (width > 0 && height > 0) return { width, height };
		}
		return undefined;
	}
	// JPEG: scan SOF markers (0xC0..0xCF excluding 0xC4/0xC8/0xCC)
	if (bytes[0] === 0xff && bytes[1] === 0xd8) {
		let offset = 2;
		while (offset + 9 < bytes.byteLength) {
			if (bytes[offset] !== 0xff) break;
			while (offset < bytes.byteLength && bytes[offset] === 0xff) offset += 1;
			if (offset >= bytes.byteLength) break;
			const marker = bytes[offset]!;
			offset += 1;
			if (marker === 0xd9 || marker === 0xda) break;
			if (offset + 2 > bytes.byteLength) break;
			const length = (bytes[offset]! << 8) | bytes[offset + 1]!;
			if (length < 2 || offset + length > bytes.byteLength) break;
			const isSof = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
			if (isSof && length >= 7) {
				const height = (bytes[offset + 3]! << 8) | bytes[offset + 4]!;
				const width = (bytes[offset + 5]! << 8) | bytes[offset + 6]!;
				if (width > 0 && height > 0) return { width, height };
				return undefined;
			}
			offset += length;
		}
	}
	return undefined;
}
