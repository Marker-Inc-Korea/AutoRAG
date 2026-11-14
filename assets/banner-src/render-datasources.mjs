// Renders datasources.html to assets/autorag-agent-datasources.png (1568x260).
import { chromium } from "playwright";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const page = await browser.newPage({ viewport: { width: 1568, height: 260 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(resolve(here, "datasources.html")).href);
await page.screenshot({ path: resolve(here, "../autorag-agent-datasources.png") });
await browser.close();
