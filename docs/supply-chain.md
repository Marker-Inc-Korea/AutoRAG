# Supply-chain gates

AutoRAG 2.0 is MIT; `legacy/` is Apache-2.0. Production dependencies must stay on the permissive SPDX allowlist in `.github/dependency-review-config.yml` (same list as `AUTORAG_SUPPLY_CHAIN_POLICY`). Copyleft GPL/AGPL/SSPL and unknown licenses fail closed.

| Gate | Tool | When it blocks |
| --- | --- | --- |
| SCA | `actions/dependency-review-action` | PRs that add high+ advisories or disallowed licenses |
| CVE | Google OSV-Scanner reusable workflows | New vulns on PRs; full scan on schedule and npm release |
| SBOM | Anchore Syft (`anchore/sbom-action`) | Missing CycloneDX/SPDX artifacts; attested on release |
| License + NOTICE | `bun scripts/supply-chain/evaluate.ts gate` | Allowlist miss or stale `NOTICE` |
| SAST | GitHub CodeQL | New codeql alerts in JS/TS and Python |
| Release | `.github/workflows/release.yml` `needs: [supply-chain]` | npm publish cannot run if the reusable supply-chain workflow failed |

Regenerate attribution with `bun scripts/supply-chain/evaluate.ts notice`. Local CycloneDX: `bun scripts/supply-chain/evaluate.ts sbom --sbom sbom.local.cdx.json`.
