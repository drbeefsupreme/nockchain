# Pitfalls Research

**Domain:** Statistical benchmark baseline and regression-attribution workflows for `nockchain-bench`
**Researched:** 2026-02-24
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Mutable Baselines That Get Silently Rewritten

**What goes wrong:**
Teams accidentally overwrite the reference baseline, so the "before" target drifts and regressions disappear.

**Why it happens:**
Baseline commands are run in overwrite mode (`--save-baseline`) during normal compare runs, and there is no immutable baseline policy.

**How to avoid:**
Treat baseline snapshots as immutable datasets: create versioned baseline IDs, compare using read-only mode (`--baseline`), and only refresh baselines in an explicit "reseed" workflow with review.

**Warning signs:**
- Baseline IDs are reused instead of appended.
- A large code change reports "no change" across most benchmarks.
- No audit trail for who refreshed a baseline and why.

**Phase to address:**
Phase 1 - Baseline data model and immutability policy.

---

### Pitfall 2: Environment Drift Masquerading as Code Regressions

**What goes wrong:**
Result deltas are driven by host variance (VM jitter, CPU governor, noisy neighbors), not by `nockchain`/`nockvm` changes.

**Why it happens:**
Benchmarks run on changing hosted environments without pinning CPU behavior, affinity, or runner class.

**How to avoid:**
Define a benchmark execution contract: pinned runner type, CPU-affinity policy, warm machine checks, and a "trusted" lane (prefer stable self-hosted or controlled hardware for release-grade baselines).

**Warning signs:**
- Same commit produces materially different medians across reruns.
- StdDev/IQR balloons for many benchmarks at once.
- Regressions appear/disappear when rerun with no code changes.

**Phase to address:**
Phase 2 - Reproducible execution environment and CI lane design.

---

### Pitfall 3: Missing Provenance (Cannot Attribute Changes)

**What goes wrong:**
A delta is detected, but root cause is unknowable because key context is absent.

**Why it happens:**
Run artifacts omit commit SHAs, benchmark config, toolchain version, CPU details, and run parameters.

**How to avoid:**
Require a provenance manifest per run: repo SHA(s), dirty/clean status, benchmark suite version, cargo profile/toolchain, OS/kernel/CPU, and statistical settings.

**Warning signs:**
- Historical files contain timings but not machine/toolchain metadata.
- Investigations rely on chat memory or PR comments.
- "Can we reproduce run X?" cannot be answered quickly.

**Phase to address:**
Phase 1 - Result schema and metadata contract.

---

### Pitfall 4: Build/Profile Inconsistency Across Baseline and Candidate

**What goes wrong:**
Teams compare numbers from different compiler/profile settings and interpret compiler/config effects as runtime regressions.

**Why it happens:**
Cargo profile, rustflags, LTO, incremental settings, or toolchain differ between baseline and candidate runs.

**How to avoid:**
Pin benchmark profile and toolchain in code, record effective build flags in artifacts, and fail runs when baseline/candidate build manifests differ.

**Warning signs:**
- Sudden global speedup/slowdown after toolchain update.
- Bench run directories built with different profiles.
- Performance shifts correlate with CI image/toolchain bumps.

**Phase to address:**
Phase 2 - Build determinism and run preflight validation.

---

### Pitfall 5: Statistical Gates Tuned for Noise, Not Decisions

**What goes wrong:**
The system floods PRs with false alarms or misses meaningful regressions because significance/noise parameters are arbitrary.

**Why it happens:**
Defaults are used without calibrating sensitivity to real workload variance; teams optimize for "passing CI" instead of decision quality.

**How to avoid:**
Calibrate thresholds from repeated no-change runs, set explicit false-positive budget, and encode severity tiers (warn vs fail) based on effect size and confidence interval overlap.

**Warning signs:**
- Frequent fail/pass flips on rerun of same commit.
- Reviewers ignore benchmark alerts as "flaky."
- Tiny deltas repeatedly fail while known large regressions slip through.

**Phase to address:**
Phase 3 - Statistical policy calibration and alert semantics.

---

### Pitfall 6: Measuring the Wrong Work (Setup/IO/One-Off Effects)

**What goes wrong:**
Reported baseline tracks harness/setup overhead or unstable IO instead of target code performance.

**Why it happens:**
Benchmark functions include setup paths, external dependencies, or mixed concerns; warmup and sampling strategy are not designed per benchmark type.

**How to avoid:**
Separate setup from measured region, classify benchmarks (micro vs macro), standardize warmup/sampling rules, and quarantine non-deterministic IO-heavy tests from regression gates.

**Warning signs:**
- Outlier rates spike for specific tests with network/filesystem activity.
- "Optimization" in setup code changes benchmark more than target function changes.
- Different harness refactors shift all benchmark numbers.

**Phase to address:**
Phase 2 - Benchmark definition standards and harness validation.

---

### Pitfall 7: History Loss in Publication/Storage Pipeline

**What goes wrong:**
Baseline history disappears or becomes incomplete, breaking longitudinal analysis.

**Why it happens:**
Teams rely on ephemeral artifacts (retention limits) or exceed GitHub Pages constraints (size/build time), with no archival strategy.

**How to avoid:**
Use dual persistence: durable canonical dataset in repo/object storage plus Pages as read-optimized publication; enforce retention/size checks in CI and roll older raw payloads into compact summaries.

**Warning signs:**
- Missing run files for older periods.
- Pages deploys timing out or failing near size limits.
- Inability to recompute trends from retained data.

**Phase to address:**
Phase 4 - Publication architecture, retention, and compaction.

---

### Pitfall 8: No Drift-Detection Controls for the Benchmarking System Itself

**What goes wrong:**
The measurement system degrades silently (runner image changes, harness bugs), and teams trust corrupted signals.

**Why it happens:**
No fixed canary benchmarks or control charts exist to detect harness/environment drift independent of product changes.

**How to avoid:**
Add sentinel benchmarks with expected stability bands, run periodic fixed-SHA control jobs, and block baseline refresh if canaries drift.

**Warning signs:**
- Broad shifts across unrelated benchmarks in same direction.
- Baseline refresh frequency increases without known performance work.
- Bench pipeline version changes with no detectable governance.

**Phase to address:**
Phase 3 - Benchmark system observability and canary controls.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reuse one baseline name forever | Simple CLI usage | No historical anchor, attribution collapse | Never |
| Benchmark only on default hosted runners | Zero infra setup | High variance and flaky regression signal | Only for non-gating smoke checks |
| Store only aggregate stats (drop raw samples) | Small files | Cannot reanalyze thresholds/statistics later | Only if raw samples are archived elsewhere |
| Manual Pages updates | Fast initial ship | Gaps, race conditions, stale publication | Only during first prototype week |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Criterion baseline CLI | Using `--save-baseline` in compare workflows | Use read-only compare (`--baseline`) in normal CI; reserve save for reviewed reseed workflow |
| GitHub-hosted runners | Treating ephemeral VMs as stable lab hardware | Separate noisy CI checks from trusted baseline lane; pin runner class and capture image metadata |
| GitHub Actions artifacts | Assuming artifacts are permanent storage | Set explicit retention policy and copy canonical data to durable store/repo |
| GitHub Pages publication | Publishing raw growing history without size controls | Publish indexed/compacted outputs and enforce size/time budgets before deploy |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Monolithic JSON history file | Publish and parse time grows each run | Partition by date/commit range and maintain summary index | Usually after months of frequent runs |
| Running full benchmark suite on every PR synchronously | CI queues, reruns, flaky failures | Split fast gate set vs scheduled full suite | As PR volume increases |
| Always-on strict fail thresholds | Developer distrust of signal | Tiered policy: warn on small effects, fail on high-confidence impactful deltas | Early, once noise is non-trivial |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting benchmark artifacts without integrity checks | Tampered datasets can poison baselines | Validate artifact digests and provenance before ingestion |
| Allowing unreviewed workflow edits to baseline-refresh path | Baseline can be maliciously rewritten | Protect workflow files and require reviewer approval for reseed jobs |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Reporting only p-values without effect size/context | Maintainers cannot prioritize action | Show effect size, confidence range, and practical threshold together |
| One giant benchmark table with no grouping | Review fatigue, missed regressions | Group by subsystem (`nockchain`/`nockvm`) and severity tier |

## "Looks Done But Isn't" Checklist

- [ ] **Baseline persistence:** Historical data is retained beyond artifact expiration windows.
- [ ] **Attribution metadata:** Every run records commit, environment, benchmark config, and statistical settings.
- [ ] **Reproducibility:** Same commit rerun has documented expected variance envelope.
- [ ] **Governance:** Baseline refresh path is explicit, reviewed, and separately auditable.
- [ ] **Publication:** Pages output can be rebuilt from canonical stored data.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Baseline overwritten | HIGH | Freeze gates, recover last trusted snapshot, replay candidate runs, and reissue baseline lineage |
| Metadata missing | HIGH | Mark affected runs non-attributable, backfill what is recoverable, and enforce schema validation |
| Environment drift discovered late | MEDIUM | Re-run reference commits in controlled lane, recalibrate thresholds, and invalidate noisy window |
| Pages history truncated | MEDIUM | Rehydrate from canonical store/artifacts, republish indexed history, add deploy guardrails |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Mutable baselines | Phase 1 | Attempted overwrite without reseed approval is blocked |
| Missing provenance | Phase 1 | Schema validator rejects runs missing required metadata |
| Environment drift | Phase 2 | Repeated same-SHA runs stay inside variance SLO |
| Build/profile mismatch | Phase 2 | Baseline/candidate manifest diff check passes |
| Wrong work measured | Phase 2 | Harness tests confirm setup excluded from timed region |
| Mis-tuned statistical gates | Phase 3 | Historical replay meets false-positive budget target |
| No drift canaries | Phase 3 | Sentinel benchmark alerts fire on induced environment change |
| History loss in publication | Phase 4 | Restore drill can regenerate Pages from canonical archive |

## Sources

- Criterion.rs analysis process and noise threshold/baseline comparison docs (official): https://bheisler.github.io/criterion.rs/book/analysis.html (MEDIUM)
- Criterion.rs CLI baseline management (`--save-baseline`, `--baseline`, `--load-baseline`): https://bheisler.github.io/criterion.rs/book/user_guide/command_line_options.html (MEDIUM)
- Criterion.rs advanced statistical configuration (sample size, significance): https://bheisler.github.io/criterion.rs/book/user_guide/advanced_configuration.html (MEDIUM)
- Criterion.rs FAQ on CI noise and false detections: https://bheisler.github.io/criterion.rs/book/faq.html (MEDIUM)
- pyperf system tuning guidance for stable benchmarks (CPU affinity/isolation, governor): https://pyperf.readthedocs.io/en/latest/system.html (MEDIUM)
- Cargo profile defaults and overrides affecting benchmark comparability: https://doc.rust-lang.org/cargo/reference/profiles.html (MEDIUM)
- GitHub-hosted runner characteristics and image update cadence: https://docs.github.com/en/actions/concepts/runners/github-hosted-runners (MEDIUM)
- GitHub Actions artifacts retention and immutability behavior in v4 examples: https://docs.github.com/en/actions/tutorials/store-and-share-data (MEDIUM)
- GitHub Pages limits (size, deploy timeout, bandwidth): https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits (MEDIUM)

---
*Pitfalls research for: statistical benchmark baseline and regression-analysis workflows*
*Researched: 2026-02-24*
