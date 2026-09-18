//! Building the desktop endpoint from whatever the audio server currently says.

use nana_core::endpoint::{EndpointId, EndpointInfo};
use nana_core::{Error, Result};

use crate::classify::classify_output;
use crate::registry::{AudioTopology, DeviceRegistry};

/// Resolves the desktop endpoint: the OS default microphone plus the OS default
/// output. Deliberately not pinned to a device, so plugging in headphones
/// switches both the output and the AEC decision without configuration.
pub struct DesktopEndpoints<R: DeviceRegistry> {
    registry: R,
}

impl<R: DeviceRegistry> DesktopEndpoints<R> {
    pub fn new(registry: R) -> Self {
        Self { registry }
    }

    pub fn topology(&self) -> Result<AudioTopology> {
        self.registry.snapshot()
    }

    pub fn current(&self) -> Result<EndpointInfo> {
        Self::from_topology(&self.registry.snapshot()?)
    }

    pub fn from_topology(topology: &AudioTopology) -> Result<EndpointInfo> {
        let input = topology.default_input().ok_or(Error::NoDevice("default input"))?;
        let output = topology.default_output().ok_or(Error::NoDevice("default output"))?;
        Ok(EndpointInfo {
            id: EndpointId("desktop".into()),
            input_name: input.description.clone(),
            output_name: output.description.clone(),
            output_class: classify_output(output),
        })
    }
}
