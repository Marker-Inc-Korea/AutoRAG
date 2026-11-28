/**
 * Datasource template names that were renamed. A config written under the old name keeps working: the factory,
 * `autorag setup`, and the finder all resolve a template through {@link canonicalDatasourceTemplate}.
 *
 * `hister` -> `nogon`: asciimoo/hister was renamed asciimoo/nogon (v0.22.0, after a trademark complaint). The
 * connection id the operator chose (`datasources.<id>`) is never rewritten, so evidence identities such as
 * `/hister/default/pages/...` stay valid.
 */
const LEGACY_TEMPLATE_ALIASES: Readonly<Record<string, string>> = { hister: "nogon" };

/** The current template name for `name`; names that were never renamed come back unchanged. */
export function canonicalDatasourceTemplate(name: string): string {
	return Object.hasOwn(LEGACY_TEMPLATE_ALIASES, name) ? (LEGACY_TEMPLATE_ALIASES[name] as string) : name;
}
