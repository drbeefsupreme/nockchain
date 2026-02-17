# SOL Run Archive

This directory stores immutable snapshots of SOL benchmark report runs.

## Layout

- `runs-manifest.json`: index metadata consumed by `index.html`
- `runs/<run_id>/sol-benchmark-transplant-report.html`
- `runs/<run_id>/sol-benchmark-transplant-report.md`
- `runs/<run_id>/sol-benchmark-transplant-memory-profiles.json`
- `runs/<run_id>/combined_summary.tsv`

## Add A New Run

1. Create `runs/<run_id>/`.
2. Copy the four artifacts into that folder.
3. Add a new object in `runs-manifest.json`.
4. Set `latest_run_id` to the new run.
5. Commit and push to the Pages source branch.
