// Renders banner.html to assets/autorag-agent-banner.png (1568x392, same canvas as n8n's banner).
import { chromium } from "playwright";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const page = await browser.newPage({ viewport: { width: 1568, height: 392 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(resolve(here, "banner.html")).href);
await page.evaluate(() => document.fonts.ready);
await page.screenshot({ path: resolve(here, "../autorag-agent-banner.png") });
await browser.close();
