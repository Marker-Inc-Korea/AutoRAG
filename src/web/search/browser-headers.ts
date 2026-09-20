/**
 * Browser-like navigation headers for credential-free search engines.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `web/search/providers/browser-headers.ts`. oh-my-pi randomizes across a
 * header-generator fingerprint pool; AutoRAG pins the deterministic Mac
 * Chrome fallback profile (the same one oh-my-pi uses for non-randomized
 * calls) so tests and diagnostics stay reproducible.
 */

// A desktop Mac Chrome navigation fingerprint.
const CHROME_FALLBACK_HEADERS: Record<string, string> = {
	Accept:
		"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
	"Accept-Encoding": "gzip, deflate, br, zstd",
	"Accept-Language": "en-US,en;q=0.9",
	"Cache-Control": "max-age=0",
	Priority: "u=0, i",
	"Sec-Ch-Ua": '"Google Chrome";v="149", "Chromium";v="149", ";Not A Brand";v="99"',
	"Sec-Ch-Ua-Mobile": "?0",
	"Sec-Ch-Ua-Platform": '"macOS"',
	"Sec-Fetch-Dest": "document",
	"Sec-Fetch-Mode": "navigate",
	"Sec-Fetch-Site": "none",
	"Sec-Fetch-User": "?1",
	"Upgrade-Insecure-Requests": "1",
	"User-Agent":
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
};

/**
 * Build a desktop navigation fingerprint for one HTTP request.
 * `randomized` is accepted for oh-my-pi call-site compatibility; the
 * deterministic profile is always returned.
 */
export function buildBrowserNavigationHeaders(_options?: { randomized?: boolean }): Record<string, string> {
	return { ...CHROME_FALLBACK_HEADERS };
}
