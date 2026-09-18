use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lang {
    Ru,
    En,
}

impl Lang {
    pub fn code(self) -> &'static str {
        match self {
            Lang::Ru => "ru",
            Lang::En => "en",
        }
    }

    pub fn parse(code: &str) -> Option<Self> {
        match code.split(['-', '_']).next()?.to_ascii_lowercase().as_str() {
            "ru" => Some(Lang::Ru),
            "en" => Some(Lang::En),
            _ => None,
        }
    }
}

/// Session-level language that only moves on a confident detection.
///
/// Whisper's per-utterance guess is unreliable on short replies ("да", "stop"),
/// so a low-confidence result keeps whatever the session already settled on.
#[derive(Debug, Clone, Copy)]
pub struct StickyLang {
    current: Lang,
    switch_threshold: f32,
}

impl StickyLang {
    pub fn new(initial: Lang) -> Self {
        Self { current: initial, switch_threshold: 0.75 }
    }

    pub fn get(&self) -> Lang {
        self.current
    }

    /// Returns true when the session language actually changed.
    pub fn observe(&mut self, detected: Lang, confidence: f32) -> bool {
        if detected != self.current && confidence >= self.switch_threshold {
            self.current = detected;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_region_tagged_codes() {
        assert_eq!(Lang::parse("ru-RU"), Some(Lang::Ru));
        assert_eq!(Lang::parse("en_US"), Some(Lang::En));
        assert_eq!(Lang::parse("de"), None);
    }

    #[test]
    fn low_confidence_does_not_switch_session_language() {
        let mut lang = StickyLang::new(Lang::Ru);
        assert!(!lang.observe(Lang::En, 0.4));
        assert_eq!(lang.get(), Lang::Ru);
        assert!(lang.observe(Lang::En, 0.9));
        assert_eq!(lang.get(), Lang::En);
    }
}
