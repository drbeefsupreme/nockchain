# Phase 04 Canonical Master Graft Execution Plan

## Metadata

- Phase: `04-master-graft-execution-plan`
- Canonical master ref policy: use `refs/remotes/upstream/master` as source of truth, with local fallback only if upstream is unavailable.
- Scope reminder: this document is execution-ready and prescribes deterministic commands for R0..R5.
- Dependency linkage source: `.planning/phases/04-master-graft-execution-plan/04-graft-dependency-matrix.tsv`

## Deterministic Sequence

## R0 Bootstrap from upstream/master

Goal: start from a fresh baseline that can be replayed deterministically.

### Preconditions

- `git remote get-url upstream` succeeds.
- Working tree is clean before branch bootstrap (`git status --short` returns empty output).
- `refs/remotes/upstream/master` resolves to a concrete SHA.

### Commands

```bash
git fetch upstream --prune
git switch --detach refs/remotes/upstream/master
BASE_SHA="$(git rev-parse HEAD)"
export GRAFT_BRANCH="bench-graft-master-$(date -u +%Y%m%d)"
git switch -C "${GRAFT_BRANCH}"
git status --short
git tag -f phase4-r0-baseline "${BASE_SHA}"
echo "R0 baseline SHA=${BASE_SHA}"
```

### Expected Output

- Branch `${GRAFT_BRANCH}` points at the same SHA as `refs/remotes/upstream/master`.
- `git status --short` is empty.
- Tag `phase4-r0-baseline` exists and resolves to `BASE_SHA`.

### Risk Notes

- If `upstream/master` is stale or unavailable, the entire graft sequence is invalid.
- Dirty tree at bootstrap leaks unrelated files into the transplant branch.

### Rollback

```bash
git reset --hard phase4-r0-baseline
git clean -fd
```

## R1 transplant staging and guardrails

Goal: reserve the transplant checkpoint for dependency-scoped graft operations.

### Preconditions

- R0 completed and tag `phase4-r0-baseline` is present.
- Matrix row coverage includes `DEP-001..DEP-009` with non-empty `execution_step_id` and `action`.
- Source bench branch `nockchain-bench-master-candidate` is fetchable.

### Commands

```bash
awk -F'\t' 'NR==1 || /^DEP-00[1-9]\t/' .planning/phases/04-master-graft-execution-plan/04-graft-dependency-matrix.tsv
git fetch upstream nockchain-bench-master-candidate:nockchain-bench-master-candidate
git restore --source nockchain-bench-master-candidate -- crates/nockchain-bench
git status --short crates/nockchain-bench
git commit -am "chore(graft): stage bench transplant skeleton"
git tag -f phase4-r1-pre-transplant HEAD
```

### Expected Output

- `crates/nockchain-bench` exists on the graft branch with candidate content staged.
- Commit history contains a single transplant commit at this checkpoint.
- Tag `phase4-r1-pre-transplant` points to the staging commit.

### Risk Notes

- Bringing extra non-bench paths from source branch introduces branch cruft.
- Skipping matrix coverage checks allows silent dependencies to bypass control plane.

### Rollback

```bash
git reset --hard phase4-r1-pre-transplant
git clean -fd
```

## R2 remove pass for branch-only dependencies

Goal: reserve removal sequence for dependencies marked remove or no-longer-needed.

### Preconditions

- R1 transplant commit exists and tag `phase4-r1-pre-transplant` resolves.
- Matrix rows for `DEP-001`, `DEP-002`, `DEP-003`, `DEP-006`, and `DEP-009` have `action=remove`.
- Replace-pass dependencies (`DEP-004`, `DEP-007`, `DEP-008`) are not edited in this step.

### Commands

```bash
git tag -f phase4-r2-pre-remove HEAD
rg -n "NOCK_PMA_PERSIST|--pma-persist|NOCK_PMA_CANDIDATE|NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE|--data-dir|is_pma_path|MemoryBucket::Pma" crates/nockchain-bench/src
# Remove-only edits: DEP-001, DEP-002, DEP-003, DEP-006, DEP-009
# - crates/nockchain-bench/src/runner/docker.rs
# - crates/nockchain-bench/src/main.rs
# - crates/nockchain-bench/src/sampler/smaps.rs
# - crates/nockchain-bench/src/sampler/buckets.rs
rg -n "NOCK_PMA_PERSIST|--pma-persist|NOCK_PMA_CANDIDATE|NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE|--data-dir|is_pma_path|MemoryBucket::Pma" crates/nockchain-bench/src && exit 1 || true
git commit -am "feat(graft): remove non-master coupling dependencies"
```

### Expected Output

- Sweep returns zero matches for remove-only dependency symbols.
- Commit contains only removal/cleanup deltas tied to `DEP-001`, `DEP-002`, `DEP-003`, `DEP-006`, `DEP-009`.
- Tag `phase4-r2-pre-remove` preserves pre-remove checkpoint.

### Risk Notes

- Removing `--data-dir` (DEP-006, medium confidence) may break runtime path expectations.
- Removing PMA sampler paths can alter observability output used by downstream scripts.

### Rollback

```bash
git reset --hard phase4-r2-pre-remove
git clean -fd
```

## R3 replace pass for master equivalents

Goal: reserve replacement sequence for dependencies with identified master alternatives.

### Preconditions

- R2 completed with zero matches for remove-only dependency sweep.
- Matrix rows for `DEP-004`, `DEP-007`, `DEP-008` contain replacement master references.
- Pinned master SHA `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c` is available for anchor checks.

### Commands

```bash
git tag -f phase4-r3-pre-replace HEAD
git show cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c:crates/nockapp-grpc/src/services/public_nockchain/v2/block_explorer.rs | rg -n "NounDecode::from_noun|HoonMapIter::from"
git show cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c:crates/nockapp/src/kernel/boot.rs | rg -n "pub save_interval|pub new"
# Replacement edits:
# - DEP-004: replace NounSpace adapters with direct decode/iterator usage.
# - DEP-007: align --save-interval handling with master boot CLI semantics.
# - DEP-008: align --new handling with master boot CLI semantics.
rg -n "NounDecode::from_noun|HoonMapIter::from|--save-interval|--new" crates/nockchain-bench/src
git commit -am "feat(graft): replace branch-only adapters with master equivalents"
```

### Expected Output

- Replacement callsites in bench code reference master-equivalent decode/iterator and CLI semantics.
- No legacy NounSpace-only adapter dependencies remain in active paths.
- Tag `phase4-r3-pre-replace` preserves rollback point for replacement pass.

### Risk Notes

- Replacement parity may regress behavior if decode/iterator semantics diverge from prior adapters.
- Incorrect CLI mapping can modify checkpoint cadence or bootstrap behavior.

### Rollback

```bash
git reset --hard phase4-r3-pre-replace
git clean -fd
```

## R4 unresolved dependency decision gate

Goal: prevent silent continuation when unresolved dependencies remain (including `DEP-005`).

### Preconditions

- R3 completed and replacement verification passed.
- Matrix still marks `DEP-005` as `action=defer` and `status=unresolved`.
- Maintainer decision owner is identified for unresolved gate sign-off.

### Commands

```bash
git tag -f phase4-r4-gate HEAD
awk -F'\t' 'NR==1 || ($1=="DEP-005")' .planning/phases/04-master-graft-execution-plan/04-graft-dependency-matrix.tsv
rg -n "^## DEP-005 Decision Gate$|Outcome A|Outcome B|Outcome C" .planning/phases/04-master-graft-execution-plan/04-master-graft-execution-plan.md
# Record selected DEP-005 outcome in both:
# 1) this runbook section "DEP-005 Decision Gate"
# 2) DEP-005 notes column in matrix
```

### Expected Output

- A single explicit DEP-005 outcome is selected and documented in both artifacts.
- Any unresolved dependency without an outcome blocks progression to R5.
- Tag `phase4-r4-gate` identifies the decision checkpoint.

### Risk Notes

- Advancing without a selected DEP-005 path creates silent semantic drift in SOL extraction behavior.
- Decision made without evidence linkage cannot be audited later.

### Rollback

```bash
git reset --hard phase4-r4-gate
git clean -fd
```

## R5 verification and closure handoff

Goal: provide final verification checkpoint and verifier/make-gate handoff contract.

### Preconditions

- R4 completed with explicit DEP-005 decision evidence.
- Matrix rows `DEP-001..DEP-009` are fully populated with action, target files, risk, rollback, and verification command.
- Checklist IDs `P001..P005` are checked; `P006..P010` remain reserved for Plan 04-03 verifier closure.

### Commands

```bash
git tag -f phase4-r5-pre-close HEAD
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} NR>1{if($h["execution_step_id"]==""||$h["action"]==""||$h["risk_note"]==""||$h["rollback_point"]==""||$h["verification_command"]==""){print "missing=" $h["dependency_id"]; bad=1}} END{exit bad}' .planning/phases/04-master-graft-execution-plan/04-graft-dependency-matrix.tsv
rg -n "^## R[0-5] |^### Preconditions$|^### Commands$|^### Expected Output$|^### Risk Notes$|^### Rollback$|^## Stop-The-Line Criteria$|^## DEP-005 Decision Gate$" .planning/phases/04-master-graft-execution-plan/04-master-graft-execution-plan.md
rg -n "^- \\[[xX]\\] P00[1-5]\\b|^- \\[ \\] P00(6|7|8|9|10)\\b" checkpoints/master_graft_plan_implementation.md
```

### Expected Output

- Verification commands exit successfully with no missing dependency control fields.
- Runbook and checklist structure satisfies Phase 4 closure prerequisites.
- Artifacts are ready for Plan 04-03 verifier wiring without schema edits.

### Risk Notes

- Skipping pre-close validation lets malformed matrix rows pass into verifier implementation.
- Mutating reserved closure IDs during this step breaks downstream automation assumptions.

### Rollback

```bash
git reset --hard phase4-r5-pre-close
git clean -fd
```

## Dependency Treatment Linkage

- Every matrix row (`DEP-001` through `DEP-009`) must map to exactly one execution step ID in `R1`-`R4`.
- `execution_step_id`, `action`, `risk_note`, `rollback_point`, and `verification_command` are required before any execution pass is considered complete.
- Provenance class from Phase 3 must be preserved through execution planning and closure evidence.

## Stop-The-Line Criteria

- Baseline is not pinned to `upstream/master` SHA.
- Any dependency action is attempted without a matrix row and rollback point.
- A required verification command is missing for an in-scope dependency row.
- An unresolved dependency proceeds beyond `R4` without explicit decision record.
- Closure checklist IDs are changed or removed from checkpoint artifacts.
- Remove-only sweep in R2 still returns non-master coupling symbols after edits.
- Replacement pass in R3 cannot cite a concrete pinned-master equivalent callsite.

## Rollback Policy

- Use nearest checkpoint rollback anchor (`R0-baseline`, `R1-pre-transplant`, `R2-pre-remove`, `R3-pre-replace`, `R4-gate`, `R5-pre-close`) based on failing step.
- Revert only the current execution span first; preserve prior validated checkpoint evidence.
- Re-run verification commands for the reverted span before resuming.
- Record rollback trigger, impacted dependency IDs, and decision outcome in execution notes.

## Unresolved Decision Gate Requirements

- Unresolved items must include dependency ID, issue summary, options considered, selected disposition, approver, and timestamp.
- No unresolved item may be closed implicitly by progress in later checkpoints.
- Decision records must be linked from both this runbook and the dependency matrix notes field.

## DEP-005 Decision Gate

`DEP-005` remains closure-blocking until one concrete outcome is selected and documented in both this file and matrix notes.

### Outcome A: Remove

- Remove `raw-transactions` extraction path entirely from bench workflows.
- Required evidence: post-change `rg -n "raw-transactions"` has no active runtime callsites in bench code.
- Risk: capability loss for mempool extraction scenarios previously dependent on this path.

### Outcome B: Replace With Master-Equivalent

- Implement an evidenced master-aligned extraction flow using pinned callsites and semantics.
- Required evidence: link to concrete master references and passing behavior parity checks.
- Risk: partial replacement can create hidden semantic drift versus legacy bench extraction.

### Outcome C: Feature-Gate (Disabled By Default)

- Keep path behind explicit opt-in gate with default disabled state.
- Required evidence: default path cannot execute unresolved extraction logic without explicit override.
- Risk: unsupported flag creep if gate semantics are not enforced in CI and runtime docs.

R5 closure is blocked unless exactly one outcome is recorded with approver and timestamp.
