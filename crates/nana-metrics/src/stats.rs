//! Percentiles over collected turns. The plan states targets as p50 and p95, so
//! those are what the tooling reports.

use crate::{Stage, TurnRecord};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Percentiles {
    pub count: usize,
    pub p50: u64,
    pub p95: u64,
    pub max: u64,
}

pub fn percentiles(values: &mut [u64]) -> Option<Percentiles> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(Percentiles {
        count: values.len(),
        p50: nearest_rank(values, 50.0),
        p95: nearest_rank(values, 95.0),
        max: *values.last().expect("non-empty"),
    })
}

/// Nearest-rank percentile: no interpolation, so every reported figure is a
/// latency that actually occurred.
fn nearest_rank(sorted: &[u64], pct: f64) -> u64 {
    let rank = (pct / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

#[derive(Debug, Clone)]
pub struct StageStats {
    pub time_to_first_audio: Option<Percentiles>,
    pub asr: Option<Percentiles>,
    pub llm_ttft: Option<Percentiles>,
    pub tts_ttfa: Option<Percentiles>,
    pub barge_in_rate: f64,
}

impl StageStats {
    pub fn collect(records: &[TurnRecord]) -> Self {
        let field = |f: fn(&TurnRecord) -> Option<u64>| {
            percentiles(&mut records.iter().filter_map(f).collect::<Vec<_>>())
        };
        let barged = records.iter().filter(|r| r.barged_in).count();
        Self {
            time_to_first_audio: field(|r| r.time_to_first_audio_ms),
            asr: field(|r| r.asr_ms),
            llm_ttft: field(|r| r.llm_ttft_ms),
            tts_ttfa: field(|r| r.tts_ttfa_ms),
            barge_in_rate: if records.is_empty() {
                0.0
            } else {
                barged as f64 / records.len() as f64
            },
        }
    }
}

/// Stage offsets from turn start, for inspecting a single turn.
pub fn stage_offset(record: &TurnRecord, stage: Stage) -> Option<u64> {
    record.stages.iter().find(|(s, _)| *s == stage).map(|(_, ms)| *ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_report_observed_values_only() {
        let p = percentiles(&mut [100, 200, 300, 400, 5000]).unwrap();
        assert_eq!(p.count, 5);
        assert_eq!(p.p50, 300);
        assert_eq!(p.p95, 5000);
        assert_eq!(p.max, 5000);
    }

    #[test]
    fn single_sample_is_its_own_percentile() {
        let p = percentiles(&mut [1450]).unwrap();
        assert_eq!((p.p50, p.p95, p.max), (1450, 1450, 1450));
    }

    #[test]
    fn empty_input_has_no_percentiles() {
        assert!(percentiles(&mut []).is_none());
    }
}
