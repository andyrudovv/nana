//! What the audio server knows about devices, independent of how we asked it.

use nana_core::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeDirection {
    Source,
    Sink,
}

/// One capture or playback device as the audio server sees it.
#[derive(Debug, Clone)]
pub struct AudioNode {
    /// Stable identifier; this is what the default-device metadata points at.
    pub name: String,
    pub description: String,
    pub direction: NodeDirection,
    pub form_factor: Option<String>,
    pub icon_name: Option<String>,
    pub profile: Option<String>,
    pub bus: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct AudioTopology {
    pub nodes: Vec<AudioNode>,
    pub default_source: Option<String>,
    pub default_sink: Option<String>,
}

impl AudioTopology {
    pub fn node(&self, name: &str) -> Option<&AudioNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    pub fn default_input(&self) -> Option<&AudioNode> {
        self.node(self.default_source.as_deref()?)
    }

    pub fn default_output(&self) -> Option<&AudioNode> {
        self.node(self.default_sink.as_deref()?)
    }

    pub fn by_direction(&self, direction: NodeDirection) -> impl Iterator<Item = &AudioNode> {
        self.nodes.iter().filter(move |n| n.direction == direction)
    }

    pub fn is_default(&self, node: &AudioNode) -> bool {
        let default = match node.direction {
            NodeDirection::Source => self.default_source.as_deref(),
            NodeDirection::Sink => self.default_sink.as_deref(),
        };
        default == Some(node.name.as_str())
    }
}

pub trait DeviceRegistry {
    fn snapshot(&self) -> Result<AudioTopology>;
}
