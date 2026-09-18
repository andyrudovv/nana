//! Per-turn timing.
//!
//! Built in from the first slice on purpose: bolting timings onto an async
//! pipeline afterwards costs more than carrying them from the start, and
//! without them latency work is guesswork.

pub mod stats;

use std::time::{Duration, Instant};

use nana_core::lang::Lang;
use serde::{Deserialize, Serialize};

pub use stats::{Percentiles, StageStats};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    WakeDetected,
    SpeechStart,
    /// Endpointing fired. Every latency figure the plan commits to is measured
    /// from here.
    SpeechEnd,
    AsrFirstPartial,
    AsrFinal,
    LlmRequest,
    LlmFirstToken,
    LlmDone,
    TtsFirstChunk,
    PlaybackStart,
    PlaybackDone,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// Grammar match answered without calling the model.
    FastPath,
    Llm,
    /// Timer or device event Nana raised on her own.
    Proactive,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub millis: u64,
    pub ok: bool,
}

/// Collects marks during a turn; [`finish`](TurnMetrics::finish) freezes it into
/// a [`TurnRecord`].
#[derive(Debug)]
pub struct TurnMetrics {
    turn_id: u64,
    session_id: u64,
    route: Route,
    origin: Instant,
    marks: Vec<(Stage, Instant)>,
    tools: Vec<ToolCall>,
    lang: Option<Lang>,
    asr_confidence: Option<f32>,
    barged_in: bool,
    watchdog_fired: bool,
}

impl TurnMetrics {
    pub fn start(session_id: u64, turn_id: u64, route: Route) -> Self {
        Self {
            turn_id,
            session_id,
            route,
            origin: Instant::now(),
            marks: Vec::new(),
            tools: Vec::new(),
            lang: None,
            asr_confidence: None,
            barged_in: false,
            watchdog_fired: false,
        }
    }

    pub fn mark(&mut self, stage: Stage) {
        self.mark_at(stage, Instant::now());
    }

    /// Records a stage that already happened, for callers holding their own
    /// timestamp (the capture thread stamps frames before the turn exists).
    pub fn mark_at(&mut self, stage: Stage, at: Instant) {
        self.marks.push((stage, at));
    }

    pub fn tool_call(&mut self, call: ToolCall) {
        self.tools.push(call);
    }

    pub fn set_transcript(&mut self, lang: Lang, confidence: f32) {
        self.lang = Some(lang);
        self.asr_confidence = Some(confidence);
    }

    pub fn set_barged_in(&mut self) {
        self.barged_in = true;
    }

    pub fn set_watchdog_fired(&mut self) {
        self.watchdog_fired = true;
    }

    fn at(&self, stage: Stage) -> Option<Instant> {
        self.marks.iter().find(|(s, _)| *s == stage).map(|(_, t)| *t)
    }

    fn between(&self, from: Stage, to: Stage) -> Option<Duration> {
        self.at(to)?.checked_duration_since(self.at(from)?)
    }

    /// The number the whole design is judged on: silence between the user
    /// finishing and Nana being audible.
    pub fn time_to_first_audio(&self) -> Option<Duration> {
        self.between(Stage::SpeechEnd, Stage::PlaybackStart)
    }

    pub fn finish(self) -> TurnRecord {
        let offset =
            |stage: Stage| self.at(stage).map(|t| t.duration_since(self.origin).as_millis() as u64);
        TurnRecord {
            turn_id: self.turn_id,
            session_id: self.session_id,
            route: self.route,
            lang: self.lang,
            asr_confidence: self.asr_confidence,
            barged_in: self.barged_in,
            watchdog_fired: self.watchdog_fired,
            time_to_first_audio_ms: self.time_to_first_audio().map(|d| d.as_millis() as u64),
            asr_ms: self.between(Stage::SpeechEnd, Stage::AsrFinal).map(|d| d.as_millis() as u64),
            llm_ttft_ms: self
                .between(Stage::LlmRequest, Stage::LlmFirstToken)
                .map(|d| d.as_millis() as u64),
            tts_ttfa_ms: self
                .between(Stage::LlmFirstToken, Stage::TtsFirstChunk)
                .map(|d| d.as_millis() as u64),
            stages: Stage::ALL.iter().filter_map(|s| offset(*s).map(|ms| (*s, ms))).collect(),
            tools: self.tools,
        }
    }
}

impl Stage {
    pub const ALL: [Stage; 11] = [
        Stage::WakeDetected,
        Stage::SpeechStart,
        Stage::SpeechEnd,
        Stage::AsrFirstPartial,
        Stage::AsrFinal,
        Stage::LlmRequest,
        Stage::LlmFirstToken,
        Stage::LlmDone,
        Stage::TtsFirstChunk,
        Stage::PlaybackStart,
        Stage::PlaybackDone,
    ];
}

/// One finished turn, ready to be written to storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    pub turn_id: u64,
    pub session_id: u64,
    pub route: Route,
    pub lang: Option<Lang>,
    pub asr_confidence: Option<f32>,
    pub barged_in: bool,
    pub watchdog_fired: bool,
    pub time_to_first_audio_ms: Option<u64>,
    pub asr_ms: Option<u64>,
    pub llm_ttft_ms: Option<u64>,
    pub tts_ttfa_ms: Option<u64>,
    /// Milliseconds from turn start, for stages that were reached.
    pub stages: Vec<(Stage, u64)>,
    pub tools: Vec<ToolCall>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_first_audio_from_the_end_of_speech() {
        let mut turn = TurnMetrics::start(1, 1, Route::Llm);
        let t0 = Instant::now();
        turn.mark_at(Stage::SpeechEnd, t0);
        turn.mark_at(Stage::PlaybackStart, t0 + Duration::from_millis(1450));
        assert_eq!(turn.time_to_first_audio(), Some(Duration::from_millis(1450)));
        assert_eq!(turn.finish().time_to_first_audio_ms, Some(1450));
    }

    #[test]
    fn unreached_stages_stay_absent_rather_than_zero() {
        let turn = TurnMetrics::start(1, 1, Route::FastPath);
        let record = turn.finish();
        assert!(record.stages.is_empty());
        assert_eq!(record.time_to_first_audio_ms, None);
        assert_eq!(record.llm_ttft_ms, None, "fast path never calls the model");
    }

    #[test]
    fn out_of_order_marks_do_not_produce_negative_spans() {
        let mut turn = TurnMetrics::start(1, 1, Route::Llm);
        let t0 = Instant::now();
        turn.mark_at(Stage::PlaybackStart, t0);
        turn.mark_at(Stage::SpeechEnd, t0 + Duration::from_millis(10));
        assert_eq!(turn.time_to_first_audio(), None);
    }

    #[test]
    fn records_serialise_for_storage() {
        let mut turn = TurnMetrics::start(7, 42, Route::Llm);
        turn.set_transcript(Lang::Ru, 0.93);
        turn.tool_call(ToolCall { name: "weather".into(), millis: 210, ok: true });
        let json = serde_json::to_string(&turn.finish()).unwrap();
        assert!(json.contains("\"route\":\"llm\""));
        assert!(json.contains("\"lang\":\"ru\""));
    }
}
