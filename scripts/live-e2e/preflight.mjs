/** Local-only service and optional datasource preflight. */

import { execFileSync } from "node:child_process";

export const DATASOURCE_BINARIES = Object.freeze([
  "katok", "discrawl", "qmd", "rclone", "mailcrawl", "jikji", "minsync",
]);

/** @typedef {"ready" | "unavailable" | "malformed"} ServiceStatus */
/** @typedef {"ready" | "degraded" | "refused"} Verdict */

function isValidEndpoint(endpoint) {
  try {
    const url = new URL(endpoint);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

function isLoopbackEndpoint(endpoint) {
  const url = new URL(endpoint);
  return url.hostname === "127.0.0.1" || url.hostname === "localhost";
}

function safeEndpoint(endpoint) {
  try {
    const url = new URL(endpoint);
    url.username = "";
    url.password = "";
    url.search = "";
    url.hash = "";
    return url.toString();
  } catch {
    return "[redacted-invalid-endpoint]";
  }
}

function laneStatus(binary, available, configured) {
  if (available) return { status: "READY" };
  return configured
    ? { status: "FAIL", reason: `${binary}-unavailable` }
    : { status: "SKIP", reason: `${binary}-not-installed` };
}

/**
 * @param {object} options
 * @param {string} [options.endpoint]
 * @param {(input: string, init?: object) => Promise<{ok: boolean, status: number, json: () => Promise<unknown>}>} [options.fetchImpl]
 * @param {Readonly<Record<string, boolean>>} [options.binaries]
 * @param {readonly string[]} [options.configuredLanes]
 * @param {string | undefined} [options.openaiKey]
 */
export async function runPreflight(options = {}) {
  const endpoint = options.endpoint ?? process.env.AUTORAG_TEI_ENDPOINT ?? "http://127.0.0.1:18080";
  const hasOpenaiKey = options.openaiKey !== undefined || Boolean(process.env.OPENAI_API_KEY || process.env.AUTORAG_OPENAI_API_KEY);
  const openaiKeyPresent = false;
  const configuredLanes = new Set(options.configuredLanes ?? process.env.AUTORAG_LIVE_E2E_LANES?.split(",").map((lane) => lane.trim()).filter(Boolean) ?? []);
  const binaries = options.binaries ?? detectBinaries();
  const lanes = Object.fromEntries(DATASOURCE_BINARIES.map((binary) => [
    binary, laneStatus(binary, binaries[binary] === true, configuredLanes.has(binary)),
  ]));
  /** @type {Record<string, {status: string, reason?: string}>} */
  const typedLanes = lanes;

  if (hasOpenaiKey) {
    return result("refused", "live-e2e-openai-egress", endpoint, openaiKeyPresent, typedLanes, { status: "unavailable" });
  }
  if (!isValidEndpoint(endpoint)) {
    return result("refused", "live-e2e-embedding-malformed", endpoint, openaiKeyPresent, typedLanes, { status: "malformed" });
  }
  if (!isLoopbackEndpoint(endpoint)) {
    return result("refused", "live-e2e-non-loopback-embedding", endpoint, openaiKeyPresent, typedLanes, { status: "malformed" });
  }

  const fetchImpl = options.fetchImpl ?? fetch;
  let embedding;
  try {
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ inputs: ["preflight"] }),
    });
    if (!response.ok) return result("refused", "live-e2e-embedding-unavailable", endpoint, openaiKeyPresent, typedLanes, { status: "unavailable" });
    const body = await response.json();
    const vector = Array.isArray(body) && Array.isArray(body[0]) ? body[0] : null;
    if (vector === null || vector.some((value) => typeof value !== "number") || vector.length !== 768) {
      return result("refused", "live-e2e-embedding-malformed", endpoint, openaiKeyPresent, typedLanes, { status: "malformed", expectedModel: "embeddinggemma:latest" });
    }
    embedding = { status: "ready", expectedModel: "embeddinggemma:latest", dimension: vector.length };
  } catch {
    return result("refused", "live-e2e-embedding-unavailable", endpoint, openaiKeyPresent, typedLanes, { status: "unavailable" });
  }

  const configuredFailures = [...configuredLanes].filter((lane) => typedLanes[lane]?.status === "FAIL");
  const verdict = configuredFailures.length > 0 ? "refused" : Object.values(typedLanes).some((lane) => lane.status === "SKIP") ? "degraded" : "ready";
  return result(verdict, verdict === "refused" ? "live-e2e-datasource-failure" : undefined, endpoint, openaiKeyPresent, typedLanes, embedding);
}

function result(verdict, code, endpoint, openaiKeyPresent, lanes, embedding) {
  return { verdict, exitCode: verdict === "refused" ? 1 : 0, ...(code ? { code } : {}), endpoint: safeEndpoint(endpoint), openaiKeyPresent, embedding, lanes };
}

function detectBinaries() {
  return Object.fromEntries(DATASOURCE_BINARIES.map((binary) => {
    try {
      execFileSync("which", [binary], { stdio: "ignore" });
      return [binary, true];
    } catch {
      return [binary, false];
    }
  }));
}
