# Security Policy

## Supported versions

- AutoRAG 2.x (`@autorag/librarian` at the repository root)
- The latest legacy Python release under `legacy/`

Older 2.x patch lines and older legacy Python releases are not patched. Please upgrade to the latest 2.x release, or to the latest legacy release if you are still on the Python AutoML tree.

## Reporting a vulnerability

Use GitHub private vulnerability reporting on this repository. Do not file a public issue for an undisclosed vulnerability.

Security reports must include a reproduction procedure and an impact scope (what an attacker can do, against which component, with what access). Reports that do not include both are not triaged as vulnerabilities. If an AI tool helped you find the issue, say so in the first sentence, reproduce it yourself, and write the report in your own words. Fabricated or copy-pasted AI security reports are not accepted; see [AI_POLICY.md](AI_POLICY.md).

## Response SLA

- **First response:** within 3 business days of a private report (Korea Standard Time, Marker Inc. business days).
- **Status updates:** at least weekly until the issue is closed, declined, or a fix is published.
- **Disclosure / embargo:** 90 days from first response, or until a patched release ships, whichever is sooner. We may ask for a longer embargo when a coordinated fix needs more time; we will not disclose a report before a patch without the reporter's consent except where we are legally required to.
- **How fixes ship:** a patched release of the affected tree (npm `@autorag/librarian` for AutoRAG 2.x, PyPI `AutoRAG` for legacy when that tree is affected) plus a GitHub Security Advisory. The advisory is the public patch-distribution record.

This SLA is a target for good-faith reports with a reproduction. We may close reports that cannot be reproduced, that describe a documented non-goal (for example remote embedding of corpus text), or that are already public.

## Secret scanning

GitHub secret scanning and push protection are enabled on this repository. Pull requests also run a gitleaks scan of the PR commit range (`.github/workflows/secret-scan.yml`). Newly introduced secrets fail the job. A one-time scan of the full git history is a follow-up; this project will not rewrite history to purge old secrets as part of a security-policy change.

## Supply chain

Dependency, license, SBOM, CVE, and release gates are documented in `docs/supply-chain.md`. Changing the project license itself is a Marker Inc. ownership decision under `GOVERNANCE.md`; these gates only enforce third-party compatibility with the current MIT + Apache-2.0 split. OpenSSF Scorecard and the OSPS Baseline mapping are in `docs/openssf-baseline.md`.
