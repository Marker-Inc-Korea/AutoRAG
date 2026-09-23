# OpenSSF Scorecard and OSPS Baseline

AutoRAG measures repository security with [OpenSSF Scorecard](https://github.com/ossf/scorecard) and maps process holes against the [OpenSSF Source Project Security Baseline](https://baseline.openssf.org/). Scorecard is a measurement, not a replacement for OSV, CodeQL, or the supply-chain gate.

The workflow is [`.github/workflows/scorecard.yml`](../.github/workflows/scorecard.yml). It runs on pushes to `main` and weekly. `publish_results: true` only publishes from the default branch of this public repository. After the first successful default-branch run, results appear at:

- `https://api.scorecard.dev/projects/github.com/Marker-Inc-Korea/AutoRAG`
- `https://api.securityscorecards.dev/projects/github.com/Marker-Inc-Korea/AutoRAG` (legacy alias)

Do not chase a 10/10. File follow-up issues only for failing checks the project intends to fix.

## OSPS Baseline mapping

| Baseline-shaped control | Where it lives | Status |
|---|---|---|
| Public license | `LICENSE` (MIT, AutoRAG 2.0) and `legacy/LICENSE` (Apache-2.0) | Present |
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
| npm provenance | `.github/workflows/publish.yml` / release path | Present |
| SAST | GitHub CodeQL default setup | Present |
| Secret scanning | GitHub secret scanning + push protection; PR-diff gitleaks in `.github/workflows/secret-scan.yml` | Present |
| Scorecard publish | `.github/workflows/scorecard.yml` | Present |
| Signed commits | not required | Gap, accepted |
| Pin every third-party Action by SHA | new Scorecard/secret-scan workflows are pinned; older workflows still use version tags | Gap, accepted; do not rewrite unrelated workflows in this pass |
| Historical secret scan of git history | not run | Follow-up; do not rewrite history |

Software SBOMs stay CycloneDX + SPDX as documented in [supply-chain.md](supply-chain.md). Scorecard does not replace those artifacts.
