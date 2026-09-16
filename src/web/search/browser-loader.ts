/**
 * Default headless-browser escalation for bot-challenged search engines.
 *
 * oh-my-pi escalates challenged engines (google/ecosia/mojeek/…) to a
 * stealth headless browser through its puppeteer browser daemon; AutoRAG
 * ships the same mitigation as a self-registering default loader: the first
 * escalation probes for a local Chrome/Chromium, launches it via
 * `puppeteer-core` (a fresh browser per escalation — no daemon), and
 * returns the rendered page. When no browser exists the degradation is
 * explicit and total: a diagnostic is recorded
 * ({@link getWebSearchBrowserDiagnostic}), no loader is installed, and the
 * provider chain continues with plain fetches. Nothing here ever throws at
 * startup or blocks it; the browser launches only on an actual challenge.
 */
import { existsSync } from "node:fs";
import type { BrowserFallbackOptions, LoadedHtmlPage, WebSearchBrowserLoader } from "./browser-page.ts";
import { SEARCH_HARD_TIMEOUT_MS } from "./providers/utils.ts";

/** Upper bound on page/browser teardown; a dead CDP session leaves close() pending forever. */
const BROWSER_TEARDOWN_TIMEOUT_MS = 5_000;

/** Minimal page surface the loader needs (structural subset of puppeteer-core's Page). */
export interface WebSearchBrowserPage {
	goto(url: string, options?: { waitUntil?: string; timeout?: number }): Promise<{ status(): number } | null>;
	content(): Promise<string>;
	url(): string;
	close(): Promise<void>;
	setViewport(viewport: { width: number; height: number }): Promise<void>;
	evaluateOnNewDocument(fn: () => void): Promise<void>;
	waitForSelector(selector: string, options?: { timeout?: number }): Promise<unknown>;
}

/** Minimal browser surface the loader needs (structural over puppeteer-core's Browser). */
export interface WebSearchBrowser {
	newPage(): Promise<WebSearchBrowserPage>;
	close(): Promise<void>;
}

/** Launch seam: production uses puppeteer-core with the detected executable; tests inject a fake. */
export type WebSearchBrowserLaunch = (executablePath: string) => Promise<WebSearchBrowser>;

export interface EnsureWebSearchBrowserLoaderOptions {
	/** Executable existence probe (default: fs.existsSync). */
	readonly exists?: (path: string) => boolean;
	/** Candidate executable paths (default: env + platform defaults). */
	readonly candidates?: readonly string[];
	/** Browser launcher (default: lazy puppeteer-core). */
	readonly launch?: WebSearchBrowserLaunch;
}

const DEFAULT_CANDIDATES_BY_PLATFORM: Readonly<Record<string, readonly string[]>> = {
	darwin: [
		"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
		"/Applications/Chromium.app/Contents/MacOS/Chromium",
		"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
		"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
	],
	linux: [
		"/usr/bin/google-chrome",
		"/usr/bin/google-chrome-stable",
		"/usr/bin/chromium",
		"/usr/bin/chromium-browser",
		"/usr/bin/microsoft-edge",
		"/snap/bin/chromium",
	],
	win32: [
		"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
		"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
		"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
	],
};

function defaultCandidates(env: NodeJS.ProcessEnv): string[] {
	const fromEnv = [env.AUTORAG_WEB_BROWSER_PATH, env.CHROME_PATH, env.PUPPETEER_EXECUTABLE_PATH]
		.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
		.map((value) => value.trim());
	return [...fromEnv, ...(DEFAULT_CANDIDATES_BY_PLATFORM[process.platform] ?? [])];
}

async function defaultLaunch(executablePath: string): Promise<WebSearchBrowser> {
	const puppeteer = await import("puppeteer-core");
	const browser = await puppeteer.launch({
		executablePath,
		headless: true,
		args: ["--no-sandbox", "--disable-blink-features=AutomationControlled"],
	});
	return browser as unknown as WebSearchBrowser;
}

let explicitLoader: WebSearchBrowserLoader | undefined;
let defaultLoader: WebSearchBrowserLoader | undefined;
let ensurePromise: Promise<boolean> | undefined;
let diagnostic: string | undefined;

/** Install (or clear, with `undefined`) an explicit headless-browser loader, taking precedence over the default. */
export function setWebSearchBrowserLoader(loader: WebSearchBrowserLoader | undefined): void {
	explicitLoader = loader;
}

/** The recorded browser-escalation diagnostic (why no default loader is installed), if any. */
export function getWebSearchBrowserDiagnostic(): string | undefined {
	return diagnostic;
}

/** Test hook: drop the explicit loader, the memoized default, and the diagnostic. */
export function resetWebSearchBrowserLoaderForTests(): void {
	explicitLoader = undefined;
	defaultLoader = undefined;
	ensurePromise = undefined;
	diagnostic = undefined;
}

async function loadWithBrowser(
	url: string,
	options: BrowserFallbackOptions,
	signal: AbortSignal,
	timeoutMs: number,
	executablePath: string,
	launch: WebSearchBrowserLaunch,
): Promise<LoadedHtmlPage> {
	const browser = await launch(executablePath);
	let page: WebSearchBrowserPage | undefined;
	try {
		page = await browser.newPage();
		// Minimal stealth: drop the automation marker challenge pages key on.
		await page.evaluateOnNewDocument(() => {
			Object.defineProperty(navigator, "webdriver", { get: () => undefined });
		});
		await page.setViewport({ width: 1280, height: 900 });
		if (options.homeUrl) {
			await page.goto(options.homeUrl, { waitUntil: "domcontentloaded", timeout: timeoutMs });
		}
		const attempts = Math.max(1, options.attempts ?? 1);
		for (let attempt = 0; attempt < attempts; attempt++) {
			if (signal.aborted) throw new DOMException("The operation was aborted.", "AbortError");
			if (attempt > 0 && options.retryDelayMs) {
				await new Promise((resolve) => setTimeout(resolve, options.retryDelayMs));
			}
			const response = await page.goto(url, { waitUntil: "domcontentloaded", timeout: timeoutMs });
			if (options.afterNavigation) await options.afterNavigation(page, signal);
			if (options.ready) {
				await page.waitForSelector(options.ready.selector, { timeout: options.ready.timeoutMs }).catch(() => null);
			}
			const loaded: LoadedHtmlPage = {
				html: await page.content(),
				status: response?.status() ?? 200,
				url: page.url(),
			};
			if (!options.shouldFallback(loaded) || attempt === attempts - 1) return loaded;
		}
		throw new Error("Browser fallback exhausted without a response");
	} finally {
		// Teardown must settle even when the caller's signal already fired;
		// bound it with a fresh deadline instead of reusing `signal`.
		if (page) {
			await Promise.race([
				page.close().catch(() => undefined),
				new Promise((resolve) => setTimeout(resolve, BROWSER_TEARDOWN_TIMEOUT_MS)),
			]);
		}
		await Promise.race([
			browser.close().catch(() => undefined),
			new Promise((resolve) => setTimeout(resolve, BROWSER_TEARDOWN_TIMEOUT_MS)),
		]);
	}
}

/**
 * Probe for a local Chrome/Chromium and install the default escalation
 * loader. Returns true when a loader is installed. Never throws: when no
 * browser exists (or puppeteer-core cannot load), a diagnostic is recorded
 * and the provider chain simply never escalates. Memoized per process.
 */
export function ensureWebSearchBrowserLoader(options: EnsureWebSearchBrowserLoaderOptions = {}): Promise<boolean> {
	ensurePromise ??= (async () => {
		if (defaultLoader) return true;
		const exists =
			options.exists ??
			((path: string): boolean => {
				try {
					return existsSync(path);
				} catch {
					return false;
				}
			});
		let executablePath: string | undefined;
		try {
			executablePath = (options.candidates ?? defaultCandidates(process.env)).find((path) => exists(path));
		} catch (error) {
			diagnostic = `web_search browser escalation disabled: Chrome/Chromium probe failed (${String(error)})`;
			return false;
		}
		if (!executablePath) {
			diagnostic =
				"web_search browser escalation disabled: no Chrome/Chromium executable found; bot-challenged engines will fall through to the next provider";
			return false;
		}
		const launch = options.launch ?? defaultLaunch;
		const path = executablePath;
		defaultLoader = (url, fallbackOptions, signal, timeoutMs) =>
			loadWithBrowser(url, fallbackOptions, signal, timeoutMs || SEARCH_HARD_TIMEOUT_MS, path, launch);
		diagnostic = undefined;
		return true;
	})();
	return ensurePromise;
}

/**
 * Resolve the loader an escalation should use: an explicitly registered
 * loader wins; otherwise the lazily-probed default. Returns undefined when
 * no browser is available (degradation already recorded as a diagnostic).
 */
export async function resolveWebSearchBrowserLoader(): Promise<WebSearchBrowserLoader | undefined> {
	if (explicitLoader) return explicitLoader;
	const installed = await ensureWebSearchBrowserLoader();
	return installed ? defaultLoader : undefined;
}

export type { BrowserFallbackOptions, LoadedHtmlPage, WebSearchBrowserLoader };
