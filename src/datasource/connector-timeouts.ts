/**
 * Timeout policy shared by CLI-backed datasource connectors.
 *
 * `sync`/`index` are not interactive. A first run over a real archive routinely
 * takes minutes — mailcrawl measured ~12 minutes for a Gmail INBOX, lazykatok
 * ~11 minutes for a ~50k-message semantic index — so a 60-second interactive
 * default truncates the step before it can finish. A timed-out step never
 * commits its cursor, so every subsequent refresh restarts from scratch instead
 * of making progress. Interactive reads (`search`, `doctor`, chunk lookups) keep
 * the short default.
 */

/**
 * Non-interactive budget for a connector `sync`/`index` step (one hour). A no-op mailcrawl `sync` of a
 * 1,000-message INBOX measured 16 to 27 minutes because mailcrawl re-reads every message on each run
 * (NomaDamas/mailcrawl#48), so 30 minutes did not hold even when nothing had changed.
 */
export const DEFAULT_CONNECTOR_SYNC_TIMEOUT_MS = 3_600_000;

/**
 * Operator-facing hint appended to a sync/index diagnostic when the connector
 * timeout killed the step. It names the override and explains why the next
 * refresh restarts rather than resumes.
 */
export function connectorSyncTimeoutHint(): string {
	return (
		"the connector.indexTimeoutMs budget cut this sync/index step short; the first " +
		"full sync/index of a large archive can take many minutes, and a timed-out step " +
		"does not commit its cursor, so the next refresh restarts from scratch. Raise " +
		"connector.indexTimeoutMs to let it finish."
	);
}
