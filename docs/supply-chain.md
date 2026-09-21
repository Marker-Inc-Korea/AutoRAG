# Supply-chain gates

Project ownership and who may change the **project** license are defined in [GOVERNANCE.md](../GOVERNANCE.md): contributions are MIT, and a license change is reserved to Marker Inc. This gate does not change that license. It keeps third-party dependencies compatible with the current distribution: AutoRAG 2.0 is MIT; `legacy/` is Apache-2.0.

Production dependencies must stay on the permissive SPDX allowlist in `.github/dependency-review-config.yml` (same list as `AUTORAG_SUPPLY_CHAIN_POLICY`). Copyleft GPL/AGPL/SSPL and unknown licenses fail closed.

| Gate | Tool | When it blocks |
| --- | --- | --- |
| SCA | `actions/dependency-review-action` | PRs that add high+ advisories or disallowed licenses |
| CVE | Google OSV-Scanner reusable workflows | New vulns on PRs; full scan on schedule and npm release |
| SBOM | Anchore Syft (`anchore/sbom-action`) | Missing CycloneDX/SPDX artifacts; attested on release |
| License + NOTICE | `bun scripts/supply-chain/evaluate.ts gate` | Allowlist miss or stale `NOTICE` |
| SAST | GitHub CodeQL | New codeql alerts in JS/TS and Python |
| Release | `.github/workflows/release.yml` `needs: [supply-chain]` | npm publish cannot run if the reusable supply-chain workflow failed |

Regenerate attribution with `bun scripts/supply-chain/evaluate.ts notice`. Local CycloneDX: `bun scripts/supply-chain/evaluate.ts sbom --sbom sbom.local.cdx.json`.

## GitHub Release assets

GitHub already attaches source zip/tar from the tag. Each `v*` release additionally uploads:

- the `npm pack` tarball (the built package, including `dist/` and `skills/`)
- `LICENSE`, `NOTICE`, and `GOVERNANCE.md`
- CycloneDX/SPDX SBOMs from the supply-chain job
- `SHA256SUMS.txt` covering those files

Stage locally after a build: `bun scripts/supply-chain/stage-release-assets.ts --out release-assets`. Do not attach `node_modules` or a second full-tree zip.

OpenSSF Scorecard publishes from [`.github/workflows/scorecard.yml`](../.github/workflows/scorecard.yml) on `main` and a weekly cron. It does not replace OSV, CodeQL, or the SBOM/license gate. The OSPS Baseline mapping and remaining accepted gaps (signed commits, Action SHA pins on older workflows, historical secret scan) are in [openssf-baseline.md](openssf-baseline.md).
