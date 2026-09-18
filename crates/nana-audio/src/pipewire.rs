//! PipeWire introspection.
//!
//! One-shot `pw-dump` snapshot. Enough for tooling and for picking the endpoint
//! at startup, but the daemon needs to *react* to the default device changing
//! mid-session, so this gets replaced by a `pipewire-rs` registry listener
//! behind the same [`DeviceRegistry`] trait.

use std::process::Command;

use nana_core::{Error, Result};
use serde_json::Value;

use crate::registry::{AudioNode, AudioTopology, DeviceRegistry, NodeDirection};

pub struct PwDump {
    binary: String,
}

impl Default for PwDump {
    fn default() -> Self {
        Self { binary: "pw-dump".into() }
    }
}

impl DeviceRegistry for PwDump {
    fn snapshot(&self) -> Result<AudioTopology> {
        let out = Command::new(&self.binary)
            .output()
            .map_err(|e| Error::Device(format!("{} not runnable: {e}", self.binary)))?;
        if !out.status.success() {
            return Err(Error::Device(format!("{} exited with {}", self.binary, out.status)));
        }
        let dump: Value = serde_json::from_slice(&out.stdout)
            .map_err(|e| Error::Device(format!("unparseable pw-dump output: {e}")))?;
        Ok(parse_topology(&dump))
    }
}

pub fn parse_topology(dump: &Value) -> AudioTopology {
    let mut topology = AudioTopology::default();
    for object in dump.as_array().map(Vec::as_slice).unwrap_or_default() {
        match object.get("type").and_then(Value::as_str) {
            Some("PipeWire:Interface:Node") => {
                if let Some(node) = parse_node(object) {
                    topology.nodes.push(node);
                }
            }
            Some("PipeWire:Interface:Metadata") => read_defaults(object, &mut topology),
            _ => {}
        }
    }
    topology
}

fn parse_node(object: &Value) -> Option<AudioNode> {
    let props = object.pointer("/info/props")?;
    // Exact match only: "Audio/Source/Virtual" and friends are monitors and
    // loopbacks, never something a person speaks into.
    let direction = match props.get("media.class").and_then(Value::as_str)? {
        "Audio/Source" => NodeDirection::Source,
        "Audio/Sink" => NodeDirection::Sink,
        _ => return None,
    };
    let name = props.get("node.name").and_then(Value::as_str)?.to_string();
    let string = |key: &str| props.get(key).and_then(Value::as_str).map(str::to_string);
    Some(AudioNode {
        description: string("node.description").unwrap_or_else(|| name.clone()),
        name,
        direction,
        form_factor: string("device.form_factor"),
        icon_name: string("device.icon-name"),
        profile: string("device.profile.name"),
        bus: string("device.bus"),
    })
}

fn read_defaults(object: &Value, topology: &mut AudioTopology) {
    if object.pointer("/props/metadata.name").and_then(Value::as_str) != Some("default") {
        return;
    }
    for entry in object.get("metadata").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
    {
        // The value is a nested object, not a bare string: {"name": "alsa_output..."}.
        let Some(name) = entry.pointer("/value/name").and_then(Value::as_str) else { continue };
        match entry.get("key").and_then(Value::as_str) {
            Some("default.audio.source") => topology.default_source = Some(name.to_string()),
            Some("default.audio.sink") => topology.default_sink = Some(name.to_string()),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMP: &str = r#"[
      {"type":"PipeWire:Interface:Node","info":{"props":{
        "media.class":"Audio/Sink","node.name":"alsa_output.hifi__Speaker__sink",
        "node.description":"Speaker","device.icon-name":"audio-card-analog",
        "device.profile.name":"HiFi: Speaker: sink","device.bus":"pci"}}},
      {"type":"PipeWire:Interface:Node","info":{"props":{
        "media.class":"Audio/Source","node.name":"alsa_input.hifi__Mic1__source",
        "node.description":"Microphone"}}},
      {"type":"PipeWire:Interface:Node","info":{"props":{
        "media.class":"Audio/Source/Virtual","node.name":"monitor.of.speaker"}}},
      {"type":"PipeWire:Interface:Metadata","props":{"metadata.name":"default"},
       "metadata":[
         {"key":"default.audio.sink","value":{"name":"alsa_output.hifi__Speaker__sink"}},
         {"key":"default.audio.source","value":{"name":"alsa_input.hifi__Mic1__source"}},
         {"key":"default.configured.audio.sink","value":{"name":"something.else"}}]}
    ]"#;

    fn topology() -> AudioTopology {
        parse_topology(&serde_json::from_str(DUMP).unwrap())
    }

    #[test]
    fn keeps_real_devices_and_drops_monitors() {
        let t = topology();
        assert_eq!(t.nodes.len(), 2);
        assert!(t.node("monitor.of.speaker").is_none());
    }

    #[test]
    fn resolves_defaults_to_nodes() {
        let t = topology();
        assert_eq!(t.default_output().unwrap().description, "Speaker");
        assert_eq!(t.default_input().unwrap().description, "Microphone");
        assert!(t.is_default(t.default_output().unwrap()));
    }

    #[test]
    fn configured_default_does_not_override_active_one() {
        // default.configured.* is the user's preference, which may not be
        // connected; default.audio.* is what is actually in use.
        assert_eq!(topology().default_sink.as_deref(), Some("alsa_output.hifi__Speaker__sink"));
    }

    #[test]
    fn node_props_are_carried_through() {
        let t = topology();
        let sink = t.default_output().unwrap();
        assert_eq!(sink.profile.as_deref(), Some("HiFi: Speaker: sink"));
        assert_eq!(sink.bus.as_deref(), Some("pci"));
    }
}
