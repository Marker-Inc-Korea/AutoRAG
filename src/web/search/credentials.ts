/**
 * Environment-variable credential resolution for web search providers.
 *
 * Replaces oh-my-pi's AuthStorage/SQLite credential broker: AutoRAG keeps
 * secrets external (environment only — never written to config files, logs,
 * or diagnostics), so a provider is "available" exactly when its documented
 * env var holds a non-empty value.
 */

/** Return the first non-empty environment value among `names`, or undefined. */
export function envCredential(...names: string[]): string | undefined {
	for (const name of names) {
		const value = process.env[name]?.trim();
		if (value) return value;
	}
	return undefined;
}
