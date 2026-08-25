export const CLAWGALLERY_DATASOURCE_ID = "clawgallery";
export const CLAWGALLERY_SOURCE_KIND = "screenshots";

export function clawGallerySourcePath(instanceId: string, imageId: string): string {
	return `/${CLAWGALLERY_SOURCE_KIND}/${instanceId}/images/${encodeURIComponent(imageId)}`;
}

export function parseClawGallerySourcePath(
	source: string,
): { readonly instanceId: string; readonly imageId: string } | undefined {
	const match = new RegExp(`^/${CLAWGALLERY_SOURCE_KIND}/([^/]+)/images/(.+)$`).exec(source);
	if (match?.[1] === undefined || match[2] === undefined) return undefined;
	return { instanceId: match[1], imageId: decodeURIComponent(match[2]) };
}
