# Canonical Compatibility Target

This document defines the single immutable `master` baseline for all compatibility findings in this project phase and subsequent inventory work.

## Target Record

- Repository: `nockchain`
- Preferred remote/ref: `upstream` / `refs/remotes/upstream/master`
- Fallback remote/ref policy: if `refs/remotes/upstream/master` is unavailable locally, use `origin` / `refs/remotes/origin/master` and record the fallback explicitly in this file before adding findings.
- Pinned SHA: `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c`
- Captured: `2026-03-03T17:08:07Z`

## Pinning Rules

- All findings MUST reference this exact Pinned SHA in `branch_context` evidence.
- During Phase 1 and later phases, no finding may compare against a moving branch name without the pinned commit.
- If the team intentionally re-pins in the future, update this file first and treat the change as a new analysis epoch.

## Enforcement Note

`refs/remotes/upstream/master` is the canonical source for this capture. The documented fallback to `refs/remotes/origin/master` exists only for local remote availability issues and must be explicit whenever used.

