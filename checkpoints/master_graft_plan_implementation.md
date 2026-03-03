# Master Graft Plan Implementation Checklist

- [x] P001 Runbook `R0..R5` now includes deterministic preconditions, command sequences, expected output, risk notes, and rollback references.
- [x] P002 Dependency matrix rows `DEP-001..DEP-009` contain concrete execution_step/action/target/master_reference/risk/rollback/verification values.
- [x] P003 Remove-only dependencies (`DEP-001`,`DEP-002`,`DEP-003`,`DEP-006`,`DEP-009`) are explicitly mapped to `R2` with auditable rollback anchors.
- [x] P004 `DEP-005` unresolved handling is closure-blocking and documents explicit `Outcome A|B|C` decision paths in the canonical runbook.
- [x] P005 Stop-the-line and pre-close verification conditions are populated and ready for Plan `04-03` verifier/make-gate wiring.

## Reserved For Plan 04-03 Closure

- [x] P006 Final verifier integration complete (`make master-graft-plan-verify` invokes `./scripts/verify_master_graft_plan.sh` and blocks on checklist drift)
- [x] P007 Closure gate confirms every dependency row `DEP-001..DEP-009` is present exactly once with concrete `exact_target_files` and `master_reference` values
- [x] P008 Closure gate confirms unresolved handling is explicit (DEP-005 decision gate + Outcome A/B/C requirements are enforced, implicit unresolved closure is rejected)
- [x] P009 Closure gate confirms every dependency row has populated `risk_note`, `rollback_point`, and `verification_command` controls
- [x] P010 Closure gate confirms one-command Phase 4 closure evidence is deterministic and auditable through verifier + make-gate enforcement
