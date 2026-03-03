# Master Graft Plan Implementation Checklist

- [x] P001 Runbook `R0..R5` now includes deterministic preconditions, command sequences, expected output, risk notes, and rollback references.
- [x] P002 Dependency matrix rows `DEP-001..DEP-009` contain concrete execution_step/action/target/master_reference/risk/rollback/verification values.
- [x] P003 Remove-only dependencies (`DEP-001`,`DEP-002`,`DEP-003`,`DEP-006`,`DEP-009`) are explicitly mapped to `R2` with auditable rollback anchors.
- [x] P004 `DEP-005` unresolved handling is closure-blocking and documents explicit `Outcome A|B|C` decision paths in the canonical runbook.
- [x] P005 Stop-the-line and pre-close verification conditions are populated and ready for Plan `04-03` verifier/make-gate wiring.

## Reserved For Plan 04-03 Closure

- [ ] P006 Final verifier integration complete (`make master-graft-verify` wired)
- [ ] P007 Closure gate confirms all dependency rows have concrete target files and refs
- [ ] P008 Closure gate confirms unresolved items have explicit approved disposition
- [ ] P009 Closure gate confirms rollback anchors are populated and executable
- [ ] P010 Closure gate confirms final handoff evidence is complete and immutable
