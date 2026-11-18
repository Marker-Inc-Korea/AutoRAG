---
title: "Supply chain and licenses"
description: "The dependency, license, SBOM, and CVE gates that protect AutoRAG releases, plus the licenses and acknowledgements they enforce."
---

# Supply chain and licenses

This page documents the gates that keep AutoRAG's dependencies compatible with its distribution, how a release is assembled and attested, what licenses ship with the project, and how the OpenSSF baseline maps onto the repository. Read it before changing a dependency, the license allowlist, or a release workflow.

AutoRAG 2.0 and `legacy/` are both Apache-2.0. This gate does not change the project license; it enforces third-party compatibility with it. Who may change the project license is an ownership decision in [`GOVERNANCE.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/GOVERNANCE.md), reserved to Marker Inc.

## Dependency and license gates

Production dependencies must stay on the permissive SPDX allowlist in `.github/dependency-review-config.yml` (the same list as `AUTORAG_SUPPLY_CHAIN_POLICY` in `scripts/supply-chain/policy.ts`). Copyleft GPL/AGPL/SSPL and unknown licenses fail closed (`unknownLicense: "deny"`), and the CVE gate fails on `high` and above (`failOnCveSeverity: "high"`).

| Gate | Tool | When it blocks |
|---|---|---|
| SCA | `actions/dependency-review-action` with `.github/dependency-review-config.yml` | PRs that add high+ advisories or disallowed licenses |
| CVE | Google OSV-Scanner reusable workflows in `.github/workflows/supply-chain.yml` | New vulns on PRs; full scan on schedule and reusable invocation |
| SBOM | Anchore Syft (`anchore/sbom-action`) | Missing CycloneDX/SPDX artifacts; attested on release |
| License and NOTICE | `bun scripts/supply-chain/evaluate.ts gate` | Allowlist miss or stale `NOTICE` |
| SAST | GitHub CodeQL default setup | New CodeQL alerts in JS/TS and Python |
| Secret scan | GitHub secret scanning + push protection; PR-diff gitleaks in `.github/workflows/secret-scan.yml` | A newly introduced secret in the PR commit range |
| Release | `.github/workflows/release.yml` `needs: [supply-chain]` | npm publish cannot run if the reusable supply-chain workflow failed |

Run the local gate with `make supply-chain` (which is `bun scripts/supply-chain/evaluate.ts gate --sbom sbom.local.cdx.json`). Regenerate the attribution file with `bun scripts/supply-chain/evaluate.ts notice`, and write a local CycloneDX SBOM with `bun scripts/supply-chain/evaluate.ts sbom --sbom sbom.local.cdx.json`.

The `NOTICE` file is generated from the production dependencies only (`package.json` `dependencies`, resolved from installed `node_modules`) plus the bundled Windows binaries; the gate compares it byte for byte against the generated text.

## GitHub Release assets

GitHub attaches source zip/tar from the tag. Each `v*` release additionally uploads:

- the `npm pack` tarball (the built package, including `dist/` and `skills/`)
- `LICENSE`, `NOTICE`, and `GOVERNANCE.md`
- the CycloneDX/SPDX SBOMs from the supply-chain job
- `SHA256SUMS.txt` covering those files

Stage them locally after a build:

```bash
bun scripts/supply-chain/stage-release-assets.ts --out release-assets
```

Do not attach `node_modules` or a second full-tree zip. The `v*` tag must equal `v<package.json version>`; the release workflow fails otherwise. Legacy Python releases use `legacy-v*` tags and `.github/workflows/publish.yml` (`uv build`/`uv publish`), which never triggers the npm release.

## SBOM and attestation

The supply-chain job produces CycloneDX and SPDX SBOMs with Syft and attaches them to the release. The release job publishes to npm through Trusted Publishing (OIDC, `id-token: write`), so the package carries npm provenance and no npm token is stored. Software SBOMs stay CycloneDX + SPDX; Scorecard does not replace them.

## Licenses and notices

Full texts live in [`NOTICE`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/NOTICE) and [`licenses/`](https://github.com/Marker-Inc-Korea/AutoRAG/tree/main/licenses).

- **AutoRAG 2.0** is released under the Apache License 2.0; **legacy Python AutoRAG** (`legacy/`) is also Apache-2.0.
- Production third-party notices and licenses are documented in `NOTICE`.
- The bundled Windows binaries keep their own licenses: Everything (MIT, plus PCRE BSD-3-Clause) in `licenses/everything-MIT-and-PCRE-BSD.txt`, and ES (MIT) in `licenses/es-MIT.txt`. Everything's source code is not public; AutoRAG redistributes the unmodified voidtools portable ZIPs.
- **FSearch / fsearch-cli** (GPL-2.0-or-later) is **not** distributed with AutoRAG: no code is copied, linked, or bundled. Users install fsearch-cli themselves and AutoRAG talks to it only through its CLI and daemon socket — separate programs, per the [FSF mere-aggregation FAQ](https://www.gnu.org/licenses/gpl-faq.html#MereAggregation).
- **Hister** (AGPL-3.0-or-later) is **not** distributed with AutoRAG: no Hister code is copied, linked, bundled, or modified. Users run an unmodified Hister release; AutoRAG talks to that separate process only over its public HTTP API (`licenses/hister-AGPL-3.0-notice.txt`). Running Hister as a service for other people makes AGPL section 13 apply to its operator.
- **Filee** (MIT) is not bundled either: AutoRAG downloads the SHA-256-pinned [`filee-cli`](https://github.com/NomaDamas/filee-cli) release into `<AUTORAG_HOME>/tools/filee-cli/` and spawns it as a separate process.

## Acknowledgements

AutoRAG stands on other open-source projects:

- **[Pi](https://github.com/earendil-works/pi-mono)** by @earendil-works — the agent framework powering AutoRAG's reasoning loop.
- **[MinSync](https://github.com/Marker-Inc-Korea/minsync)** — incremental Change Data Capture chunking, local sparse/vector indexing, and the Qwen3-Embedding-0.6B embedding engine.
- **[Jikji](https://github.com/NomaDamas/jikji)** by NomaDamas — find-first local document discovery.
- **[Everything](https://www.voidtools.com/)** and **[ES](https://github.com/voidtools/ES)** by voidtools — Windows file-name indexing and its CLI, bundled under the MIT License.
- **[FSearch](https://github.com/cboxdoerfer/fsearch)** and **[fsearch-cli](https://github.com/NomaDamas/fsearch-mac)** — Everything-style file-name search for macOS/Linux (GPL-2.0-or-later), never bundled or linked.
- **[Hister](https://github.com/asciimoo/hister)** — self-hosted browser web-history search (AGPL-3.0-or-later), never bundled, linked, copied, or modified.
- **[dupey](https://github.com/NomaDamas/dupey)** by NomaDamas — duplicate and near-duplicate document family detection.
- **[Filee](https://github.com/KnifeLemon/Filee)** by KnifeLemon — offline conversion of 240+ file formats (MIT), downloaded as the separate `filee-cli` process.
- **Federated CLI authors** — [`lazykatok`](https://github.com/changeroa/lazykatok), [`discrawl`](https://github.com/openclaw/discrawl), [`mailcrawl`](https://github.com/NomaDamas/mailcrawl), [`wacrawl`](https://github.com/openclaw/wacrawl), [`telecrawl`](https://github.com/openclaw/telecrawl), [`slacrawl`](https://github.com/openclaw/slacrawl), and [`notcrawl`](https://github.com/openclaw/notcrawl).
- **[kordoc](https://github.com/chrisryugj/kordoc)** by chrisryugj — HWP/HWPX/HWPML, PDF, DOCX, and XLSX parsing with nested-table fidelity, used as AutoRAG's default document parser.

## OpenSSF Scorecard and OSPS baseline

AutoRAG measures repository security with [OpenSSF Scorecard](https://github.com/ossf/scorecard) and maps process gaps against the [OpenSSF Source Project Security Baseline](https://baseline.openssf.org/). Scorecard is a measurement, not a replacement for OSV, CodeQL, or the supply-chain gate. The workflow is `.github/workflows/scorecard.yml`; it runs on pushes to `main` and weekly, and `publish_results: true` publishes only from the default branch of this public repository. Do not chase a 10/10; file follow-up issues only for failing checks the project intends to fix.

Mapping of baseline-shaped controls to where they live:

| Baseline-shaped control | Where it lives | Status |
|---|---|---|
| Public license | `LICENSE` and `legacy/LICENSE` (both Apache-2.0) | Present |
| Security policy and vuln inbox | `SECURITY.md` (private GitHub reporting + SLA) | Present |
| Code of conduct | `CODE_OF_CONDUCT.md` | Present |
| Contribution and DCO | `CONTRIBUTING.md`, `.github/workflows/dco.yml` | Present |
| Governance and maintainers | `GOVERNANCE.md`, `MAINTAINERS` | Present |
| AI inbound rule | `AI_POLICY.md` | Present |
| Branch protection / required review | rulesets `main-ci-required` and `main-review-required` | Present |
| CODEOWNERS | `.github/CODEOWNERS` | Present |
| CI on every PR | `.github/workflows/ci.yml` (required `build` check) | Present |
| Dependency and license review | `.github/workflows/supply-chain.yml`, `.github/dependency-review-config.yml` | Present |
| CVE scanning | OSV-Scanner reusable workflows in `supply-chain.yml` | Present |
| SBOM + attestation | Syft CycloneDX/SPDX in `supply-chain.yml`; attested on release | Present |
| npm provenance | release path in `.github/workflows/release.yml` | Present |
| SAST | GitHub CodeQL default setup | Present |
| Secret scanning | GitHub secret scanning + push protection; PR-diff gitleaks in `.github/workflows/secret-scan.yml` | Present |
| Scorecard publish | `.github/workflows/scorecard.yml` | Present |
| Signed commits | not required | Gap, accepted |
| Pin every third-party Action by SHA | new Scorecard/secret-scan workflows are pinned; older workflows still use version tags | Gap, accepted |
| Historical secret scan of git history | not run; the project will not rewrite history to purge old secrets | Follow-up |

## Related pages

- [Development workflow](development.md) for the guardrails and the checks a pull request runs.
- [Architecture](architecture.md) for where state lands on disk.
- [`SECURITY.md`](https://github.com/Marker-Inc-Korea/AutoRAG/blob/main/SECURITY.md) for reporting a vulnerability.
