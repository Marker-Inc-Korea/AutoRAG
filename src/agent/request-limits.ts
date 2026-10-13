import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ImageContent, ModelInputLimits, TextContent } from "@earendil-works/pi-ai";

/** Stands in for an image dropped from a model request by {@link fitRequestLimits}. */
export const IMAGE_OMITTED_NOTE =
	"[Image omitted from this request to stay within the model provider's request limits. Read the file again if you still need to see it.]";

interface ImageLocation {
	readonly message: number;
	readonly block: number;
	readonly bytes: number;
}

const NOTE_BYTES = Buffer.byteLength(JSON.stringify({ type: "text", text: IMAGE_OMITTED_NOTE }));

/**
 * Fits a model request into the provider's declared `inputLimits`.
 *
 * pi keeps every image a tool returned in the transcript and resends all of
 * them on each later turn, but it never enforces `maxRequestBytes` or
 * `images.maxPerRequest`. A search that reads a handful of photos therefore
 * grows past a provider's body limit and the request fails outright (the hosted
 * AutoRAG API answers HTTP 413). Images are what grows the body, so the oldest
 * ones are replaced with {@link IMAGE_OMITTED_NOTE} until the request fits; the
 * newest images, which the model is looking at, go last. Text is never touched.
 *
 * The size is the serialized transcript, a close lower bound of the provider
 * body, so `maxRequestBytes` should keep headroom below the hard limit.
 * Returns undefined when the request already fits; never mutates `messages`.
 */
export function fitRequestLimits(
	messages: readonly AgentMessage[],
	limits: ModelInputLimits | undefined,
): AgentMessage[] | undefined {
	const maxBytes = limits?.maxRequestBytes;
	const maxImages = limits?.images?.maxPerRequest;
	if (maxBytes === undefined && maxImages === undefined) return undefined;
	const images = imageLocations(messages);
	if (images.length === 0) return undefined;

	let keptImages = images.length;
	let bytes = maxBytes === undefined ? 0 : Buffer.byteLength(JSON.stringify(messages));
	const dropped: ImageLocation[] = [];
	for (const image of images) {
		const overCount = maxImages !== undefined && keptImages > maxImages;
		const overBytes = maxBytes !== undefined && bytes > maxBytes;
		if (!overCount && !overBytes) break;
		dropped.push(image);
		keptImages--;
		bytes -= image.bytes - NOTE_BYTES;
	}
	if (dropped.length === 0) return undefined;

	const byMessage = new Map<number, Set<number>>();
	for (const { message, block } of dropped) {
		const blocks = byMessage.get(message) ?? new Set<number>();
		blocks.add(block);
		byMessage.set(message, blocks);
	}
	return messages.map((message, index) => {
		const blocks = byMessage.get(index);
		if (blocks === undefined) return message;
		const content = (message as { content: (TextContent | ImageContent)[] }).content.map((part, block) =>
			blocks.has(block) ? ({ type: "text", text: IMAGE_OMITTED_NOTE } satisfies TextContent) : part,
		);
		return { ...message, content } as AgentMessage;
	});
}

/** Every image block in the transcript, oldest first. */
function imageLocations(messages: readonly AgentMessage[]): ImageLocation[] {
	const locations: ImageLocation[] = [];
	messages.forEach((message, index) => {
		const content = (message as { content?: unknown }).content;
		if (!Array.isArray(content)) return;
		content.forEach((part: unknown, block) => {
			if (!isImage(part)) return;
			locations.push({ message: index, block, bytes: Buffer.byteLength(JSON.stringify(part)) });
		});
	});
	return locations;
}

function isImage(part: unknown): part is ImageContent {
	return typeof part === "object" && part !== null && (part as { type?: unknown }).type === "image";
}
