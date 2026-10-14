# Security Policy

## Supported Versions

| Version | Supported          | Python Versions | Status |
| :--- | :---: | :---: | :---: |
| `0.2.x` | :white_check_mark: | 3.10, 3.11, 3.12 | Current Active Release |
| `0.1.x` | :white_check_mark: | 3.10, 3.11, 3.12 | Supported |

---

## Reporting a Security Concern or Vulnerability

We welcome security reports, rule contributions, and patches from the open-source community.

To report a vulnerability or propose a security enhancement:

1. **GitHub Security Advisory (Recommended for sensitive vulnerabilities):**
   - Submit a report privately via [GitHub Security Advisories](https://github.com/pradeepkumar-ai-byte/VaxGuard/security/advisories/new).
   - This ensures findings are reviewed and patched in private prior to public disclosure.

2. **GitHub Pull Requests & Issues (For heuristics, bugfixes, and false-positive reports):**
   - If you encounter a heuristic edge-case, a false-positive scenario, or wish to contribute a new Colang rule or integration adapter, open a [GitHub Issue](https://github.com/pradeepkumar-ai-byte/VaxGuard/issues) or submit a [Pull Request](https://github.com/pradeepkumar-ai-byte/VaxGuard/pulls).
   - All contributions must pass the full automated CI test suite (`pytest -v`).

### Review & Remediation SLA
- **Initial Triage:** Within 24 hours of GitHub submission.
- **Verification & Quarantine Run:** Within 48 hours using the VaxGuard Inoculation Lab proving ground.
- **Release:** Tested against 1,424+ automated regression suites and deployed via GitHub Actions.
