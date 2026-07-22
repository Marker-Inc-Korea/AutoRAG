export function normalizeVirtualPath(value: string): string {
	const normalized = value.replace(/\\/g, "/").replace(/\/+/g, "/").trim();
	if (normalized.length === 0 || normalized === "/") return "/";
	const withLeadingSlash = normalized.startsWith("/") ? normalized : `/${normalized}`;
	return withLeadingSlash.length > 1 ? withLeadingSlash.replace(/\/+$/g, "") : withLeadingSlash;
}

export function normalizeVirtualPathScope(scope: string | undefined): string | undefined {
	if (scope === undefined) return undefined;
	const normalized = normalizeVirtualPath(scope);
	return normalized === "/" ? undefined : normalized;
}

export function matchesVirtualPathScope(virtualPath: string, scope: string | undefined): boolean {
	const normalizedScope = normalizeVirtualPathScope(scope);
	if (normalizedScope === undefined) return true;
	return virtualPathScopeToRegExp(normalizedScope).test(normalizeVirtualPath(virtualPath));
}

export function virtualPathScopeToRegExp(scope: string): RegExp {
	const normalized = normalizeVirtualPath(scope);
	const scoped = hasGlob(normalized) || looksLikeFileScope(normalized) ? normalized : `${normalized}/**`;
	const pattern = scoped
		.split("/")
		.map((segment, index) => {
			if (index === 0) return "";
			if (segment === "**") return "(?:.*)";
			return escapeRegExp(segment).replace(/\\\*/g, "[^/]*");
		})
		.join("/")
		.replace(/\/\(\?:\.\*\)$/u, "(?:/.*)?");
	return new RegExp(`^${pattern}$`);
}

function hasGlob(scope: string): boolean {
	return scope.includes("*");
}

function looksLikeFileScope(scope: string): boolean {
	const finalSegment = scope.split("/").at(-1) ?? "";
	return /\.[^/.]+$/u.test(finalSegment);
}

function escapeRegExp(literal: string): string {
	return literal.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
