use super::ingest::CombinedSummaryRow;
use super::model::{BaselineKey, BaselinePolicy};

pub fn select_baseline_rows<'a>(
    rows: &'a [CombinedSummaryRow],
    key: &BaselineKey,
    policy: BaselinePolicy,
) -> Vec<&'a CombinedSummaryRow> {
    let mut selected: Vec<&CombinedSummaryRow> = rows
        .iter()
        .filter(|row| row.env == key.env && row.fixture == key.fixture)
        .filter(|row| match &key.branch {
            Some(branch) => row.branch == *branch,
            None => true,
        })
        .filter(|row| row.exit_status == 0 && row.failed_pokes == 0.0)
        .collect();

    if selected.len() > policy.window_runs {
        selected.truncate(policy.window_runs);
    }

    selected
}

pub fn has_sufficient_baseline(rows: &[&CombinedSummaryRow], policy: BaselinePolicy) -> bool {
    rows.len() >= policy.min_samples
}

pub fn select_baseline_rows_with_fallback<'a>(
    rows: &'a [CombinedSummaryRow],
    key: &BaselineKey,
    policy: BaselinePolicy,
) -> (BaselineKey, Vec<&'a CombinedSummaryRow>) {
    let primary = select_baseline_rows(rows, key, policy);
    if has_sufficient_baseline(&primary, policy) {
        return (key.clone(), primary);
    }

    let fallback_key = BaselineKey {
        env: key.env.clone(),
        fixture: key.fixture.clone(),
        branch: None,
    };
    let fallback = select_baseline_rows(rows, &fallback_key, policy);
    (fallback_key, fallback)
}
