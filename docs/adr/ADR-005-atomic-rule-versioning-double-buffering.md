# ADR-005: Atomic Rule Versioning and Double-Buffered Hot-Reload

## Status
Accepted

## Context
Enterprise guardrail services operate in 24/7 environments where rules and policy definitions must be updated dynamically to counter zero-day jailbreaks or comply with revised governance directives. Naive hot-reloading approaches (e.g. modifying active pattern arrays in-place while requests are being evaluated concurrently) cause race conditions, partial rule application, dropped requests, or temporary locking bottlenecks on the critical evaluation path.

## Decision
VaxGuard adopts an **Atomic Double-Buffered Hot-Reload Architecture** (`RuleLoader`):
1. **Isolated Stage Compilation**:
   - When a reload is triggered (via API, file watcher, or CLI command), all YAML files in `data/rules/` are parsed and validated in a background stage buffer.
   - If any syntax error, malformed regex, or missing required field is detected, the operation aborts with a descriptive error; the running engine remains completely unaffected.
2. **Cryptographic Fingerprinting**:
   - A deterministic SHA-256 hash is computed over all canonical rule identifiers, patterns, and thresholds.
   - The compiled rules are bundled into an immutable `RuleSetVersion` container tagged with the fingerprint and timestamp.
3. **Atomic Pointer Swap**:
   - The active reference in `TriHybridMatcher` is swapped atomically.
   - Ongoing evaluations on previous worker threads finish cleanly against the prior immutable version snapshot without blocking.
4. **Version History & Instant Rollback**:
   - Prior `RuleSetVersion` instances are retained in an in-memory ring buffer.
   - An administrator or automated canary mechanism can invoke `rollback_to_version(fingerprint)` to restore a known-good configuration in under 1 millisecond.

## Consequences
### Positive
- **Zero-Downtime Rule Updates**: New security rules deploy immediately without process restarts or dropped HTTP requests.
- **Lock-Free Hot Path**: Query evaluation threads read an immutable reference pointer with zero lock contention.
- **Rollback Safety**: Instantaneous reversion to previous rule versions upon canary alert or operational regression.

### Negative / Trade-offs
- Memory consumption temporarily doubles during the reload compilation phase as two complete rule sets exist concurrently in memory before garbage collection cleans up unreferenced versions.
