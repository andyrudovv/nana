//! Deciding whether Nana can hear herself through a given output.
//!
//! The answer switches AEC on or off, so a wrong guess towards headphones makes
//! her interrupt herself. Every ambiguous case therefore resolves to speakers.

use nana_core::endpoint::OutputClass;

use crate::registry::AudioNode;

const HEADPHONE_WORDS: [&str; 8] =
    ["headphone", "headset", "earbud", "earphone", "airpod", "hands-free", "наушник", "гарнитур"];
const SPEAKER_WORDS: [&str; 6] =
    ["speaker", "hdmi", "displayport", "line out", "динамик", "колонк"];

pub fn classify_output(node: &AudioNode) -> OutputClass {
    // Set by PipeWire for USB and Bluetooth devices that declare themselves.
    if let Some(form) = node.form_factor.as_deref() {
        match form.to_ascii_lowercase().as_str() {
            "headset" | "headphone" | "headphones" | "earpiece" => return OutputClass::Headphones,
            "speaker" | "internal" | "hifi" | "tv" => return OutputClass::Speakers,
            _ => {}
        }
    }
    if let Some(icon) = node.icon_name.as_deref() {
        if let Some(class) = match_words(icon) {
            return class;
        }
    }
    // Built-in cards expose the jack as a separate profile, so the port name in
    // node.name or the profile carries the answer: HiFi__Headphones__sink.
    node.profile
        .as_deref()
        .and_then(match_words)
        .or_else(|| match_words(&node.name))
        .or_else(|| match_words(&node.description))
        .unwrap_or(OutputClass::Unknown)
}

fn match_words(text: &str) -> Option<OutputClass> {
    let text = text.to_ascii_lowercase();
    if HEADPHONE_WORDS.iter().any(|w| text.contains(w)) {
        Some(OutputClass::Headphones)
    } else if SPEAKER_WORDS.iter().any(|w| text.contains(w)) {
        Some(OutputClass::Speakers)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::NodeDirection;
    use nana_core::endpoint::AecMode;

    fn node(name: &str, profile: Option<&str>, form: Option<&str>) -> AudioNode {
        AudioNode {
            name: name.into(),
            description: name.into(),
            direction: NodeDirection::Sink,
            form_factor: form.map(str::to_string),
            icon_name: None,
            profile: profile.map(str::to_string),
            bus: None,
        }
    }

    #[test]
    fn form_factor_wins_when_the_device_declares_itself() {
        let n = node("bluez_output.AC_12", None, Some("headset"));
        assert_eq!(classify_output(&n), OutputClass::Headphones);
    }

    #[test]
    fn builtin_jack_is_recognised_through_the_profile_name() {
        let n = node(
            "alsa_output.pci-0000_00_1f.3.HiFi__Headphones__sink",
            Some("HiFi: Headphones: sink"),
            None,
        );
        assert_eq!(classify_output(&n), OutputClass::Headphones);
    }

    #[test]
    fn builtin_speaker_and_hdmi_are_speakers() {
        assert_eq!(
            classify_output(&node("x.HiFi__Speaker__sink", Some("HiFi: Speaker: sink"), None)),
            OutputClass::Speakers
        );
        assert_eq!(
            classify_output(&node("x.HiFi__HDMI3__sink", Some("HiFi: HDMI3: sink"), None)),
            OutputClass::Speakers
        );
    }

    #[test]
    fn nameless_device_keeps_aec_running() {
        let n = node("alsa_output.usb-0d8c_USB_Audio-00.analog-stereo", None, None);
        assert_eq!(classify_output(&n), OutputClass::Unknown);
        assert_eq!(classify_output(&n).aec_mode(), AecMode::Render);
    }

    #[test]
    fn bluetooth_bus_alone_never_disables_aec() {
        let mut n = node("bluez_output.speaker", None, None);
        n.bus = Some("bluetooth".into());
        assert_eq!(classify_output(&n).aec_mode(), AecMode::Render);
    }
}
