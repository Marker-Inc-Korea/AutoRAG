import { createCrawlerManagedCliProvider } from "../datasource/crawler-managed-config.ts";
import { createRcloneManagedCliProvider } from "../datasource/skills/cloud-drive/rclone-managed-config.ts";
import { createDiscrawlManagedCliProvider } from "../datasource/skills/discrawl/config.ts";
import { createHimalayaManagedCliProvider } from "../datasource/skills/gmail/himalaya-managed-config.ts";
import { createKatokManagedCliProvider } from "../datasource/skills/katok/config.ts";
import { createMailcrawlManagedCliProvider } from "../datasource/skills/mailcrawl/config.ts";
import { createQmdManagedCliProvider } from "../datasource/skills/obsidian/config.ts";
import { ManagedCliConfigManager, ManagedCliRegistry } from "./managed-cli-config.ts";

export function createManagedCliRuntime(
	workspace: string,
	options: { readonly mailcrawlBinaryPaths?: readonly string[] } = {},
): {
	readonly registry: ManagedCliRegistry;
	readonly manager: ManagedCliConfigManager;
} {
	const registry = new ManagedCliRegistry();
	for (const provider of [
		createDiscrawlManagedCliProvider(),
		createKatokManagedCliProvider(),
		createCrawlerManagedCliProvider("wacrawl"),
		createCrawlerManagedCliProvider("telecrawl"),
		createCrawlerManagedCliProvider("slacrawl"),
		createCrawlerManagedCliProvider("notcrawl"),
		createQmdManagedCliProvider(),
		createRcloneManagedCliProvider(),
		createHimalayaManagedCliProvider(),
		createMailcrawlManagedCliProvider(undefined, options.mailcrawlBinaryPaths),
	]) {
		registry.register(provider);
	}
	return { registry, manager: new ManagedCliConfigManager({ workspace, registry }) };
}
