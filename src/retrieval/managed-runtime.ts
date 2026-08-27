import { ManagedCliConfigManager, ManagedCliRegistry } from "../cli/managed-cli-config.ts";
import { createJikjiManagedCliProvider } from "../jikji/managed-config.ts";
import { createMinSyncManagedCliProvider } from "../minsync/managed-config.ts";

/**
 * Managed execution boundary for retrieval-owned command-line engines.
 *
 * This is intentionally separate from datasource CLI configuration: these
 * tools own retrieval/index state rather than representing a datasource.
 */
export interface ManagedRetrievalRuntimeOptions {
	readonly minSync?: boolean;
	readonly minSyncBinaryPath?: string;
	readonly jikji?: boolean;
	readonly jikjiBinaryPath?: string;
}

export class ManagedRetrievalRuntime {
	readonly registry: ManagedCliRegistry;
	readonly manager: ManagedCliConfigManager;

	constructor(workspace: string, options: ManagedRetrievalRuntimeOptions = {}) {
		this.registry = new ManagedCliRegistry();
		if (options.minSync !== false) {
			this.registry.register(createMinSyncManagedCliProvider(options.minSyncBinaryPath));
		}
		if (options.jikji === true) {
			this.registry.register(createJikjiManagedCliProvider(options.jikjiBinaryPath));
		}
		this.manager = new ManagedCliConfigManager({ workspace, registry: this.registry });
	}
}

export function createManagedRetrievalRuntime(
	workspace: string,
	options: ManagedRetrievalRuntimeOptions = {},
): ManagedRetrievalRuntime {
	return new ManagedRetrievalRuntime(workspace, options);
}
