pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

pub fn mad(values: &[f64], center: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut deviations: Vec<f64> = values.iter().map(|v| (v - center).abs()).collect();
    median(&mut deviations)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    pub low: f64,
    pub high: f64,
}

pub fn bootstrap_median_ci(
    values: &[f64],
    iterations: usize,
    alpha: f64,
    seed: u64,
) -> Option<ConfidenceInterval> {
    if values.is_empty() || iterations == 0 {
        return None;
    }
    if values.len() == 1 {
        return Some(ConfidenceInterval {
            low: values[0],
            high: values[0],
        });
    }

    let alpha = alpha.clamp(0.0001, 0.9999);
    let mut rng = seed.max(1);
    let mut medians = Vec::with_capacity(iterations);
    let n = values.len();
    for _ in 0..iterations {
        let mut sample = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = (next_u64(&mut rng) as usize) % n;
            sample.push(values[idx]);
        }
        if let Some(m) = median(&mut sample) {
            medians.push(m);
        }
    }
    if medians.is_empty() {
        return None;
    }
    medians.sort_by(f64::total_cmp);
    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - lo_q;
    let lo_idx = ((medians.len() as f64 * lo_q).floor() as usize).min(medians.len() - 1);
    let hi_idx = ((medians.len() as f64 * hi_q).floor() as usize)
        .min(medians.len() - 1)
        .max(lo_idx);
    Some(ConfidenceInterval {
        low: medians[lo_idx],
        high: medians[hi_idx],
    })
}

fn next_u64(state: &mut u64) -> u64 {
    // xorshift64*
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

#[cfg(test)]
mod tests {
    use super::{bootstrap_median_ci, mad, median};

    #[test]
    fn median_and_mad_handle_outliers() {
        let mut values = vec![10.0, 10.5, 11.0, 1000.0];
        let center = median(&mut values).expect("median");
        let spread = mad(&values, center).expect("mad");
        assert!((center - 10.75).abs() < 1e-9);
        assert!(spread < 1.0);
    }

    #[test]
    fn bootstrap_ci_is_deterministic() {
        let values = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let ci1 = bootstrap_median_ci(&values, 500, 0.05, 42).expect("ci1");
        let ci2 = bootstrap_median_ci(&values, 500, 0.05, 42).expect("ci2");
        assert_eq!(ci1, ci2);
        assert!(ci1.low <= 12.0);
        assert!(ci1.high >= 12.0);
    }

    #[test]
    fn bootstrap_handles_tiny_samples() {
        let values = vec![7.0];
        let ci = bootstrap_median_ci(&values, 100, 0.05, 9).expect("ci");
        assert_eq!(ci.low, 7.0);
        assert_eq!(ci.high, 7.0);
    }
}
