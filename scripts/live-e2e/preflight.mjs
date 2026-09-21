/** Local-only gateway and optional datasource preflight. */

import { execFileSync } from "node:child_process";

export const DATASOURCE_BINARIES = Object.freeze([
  "discrawl", "qmd", "rclone", "mailcrawl", "jikji", "minsync",
]);

const PROFILE_ID = "qwen3-embedding-0.6b";
const EXPECTED_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf";
const EXPECTED_DIMENSION = 1024;

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
    url.username = ""; url.password = ""; url.search = ""; url.hash = "";
    return url.toString();
  } catch { return "[redacted-invalid-endpoint]"; }
}
function laneStatus(binary, available, configured) {
  if (available) return { status: "READY" };
  return configured ? { status: "FAIL", reason: `${binary}-unavailable` } : { status: "SKIP", reason: `${binary}-not-installed` };
}

/** @param {object} options */
export async function runPreflight(options = {}) {
  const endpoint = options.endpoint ?? process.env.AUTORAG_GATEWAY_ENDPOINT ?? "http://127.0.0.1:0";
  const hasOpenaiKey = options.openaiKey !== undefined || Boolean(process.env.OPENAI_API_KEY || process.env.AUTORAG_OPENAI_API_KEY);
  const configuredLanes = new Set(options.configuredLanes ?? process.env.AUTORAG_LIVE_E2E_LANES?.split(",").map((lane) => lane.trim()).filter(Boolean) ?? []);
  const binaries = options.binaries ?? detectBinaries();
  const typedLanes = Object.fromEntries(DATASOURCE_BINARIES.map((binary) => [binary, laneStatus(binary, binaries[binary] === true, configuredLanes.has(binary))]));
  if (hasOpenaiKey) return result("refused", "live-e2e-openai-egress", endpoint, false, typedLanes, { status: "unavailable" });
  if (!isValidEndpoint(endpoint)) return result("refused", "live-e2e-gateway-malformed", endpoint, false, typedLanes, { status: "malformed" });
  if (!isLoopbackEndpoint(endpoint)) return result("refused", "live-e2e-non-loopback-gateway", endpoint, false, typedLanes, { status: "malformed" });

  let embedding;
  try {
    const fetchImpl = options.fetchImpl ?? fetch;
    const health = await fetchImpl(new URL("/healthz", endpoint).toString(), { method: "GET" });
    if (!health.ok) return result("refused", "live-e2e-gateway-unavailable", endpoint, false, typedLanes, { status: "unavailable" });
    const body = await health.json();
    if (!body || body.status !== "ok" || body.profileId !== PROFILE_ID || body.model !== EXPECTED_MODEL || body.dimension !== EXPECTED_DIMENSION) {
      return result("refused", "live-e2e-gateway-malformed", endpoint, false, typedLanes, { status: "malformed", expectedModel: EXPECTED_MODEL, expectedDimension: EXPECTED_DIMENSION });
    }
    embedding = { status: "ready", profileId: PROFILE_ID, expectedModel: EXPECTED_MODEL, dimension: body.dimension };
  } catch {
    return result("refused", "live-e2e-gateway-unavailable", endpoint, false, typedLanes, { status: "unavailable" });
  }
  const configuredFailures = [...configuredLanes].filter((lane) => typedLanes[lane]?.status === "FAIL");
  const verdict = configuredFailures.length > 0 ? "refused" : Object.values(typedLanes).some((lane) => lane.status === "SKIP") ? "degraded" : "ready";
  return result(verdict, verdict === "refused" ? "live-e2e-datasource-failure" : undefined, endpoint, false, typedLanes, embedding);
}
function result(verdict, code, endpoint, openaiKeyPresent, lanes, embedding) {
  return { verdict, exitCode: verdict === "refused" ? 1 : 0, ...(code ? { code } : {}), endpoint: safeEndpoint(endpoint), openaiKeyPresent, embedding, lanes };
}
function detectBinaries() {
  return Object.fromEntries(DATASOURCE_BINARIES.map((binary) => {
    try { execFileSync("which", [binary], { stdio: "ignore" }); return [binary, true]; }
    catch { return [binary, false]; }
  }));
}
