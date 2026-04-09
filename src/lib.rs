//! cuda-intelligence — GPU-accelerated frozen intelligence toolchain
//!
//! Rust implementation of mask-locked inference chip design tools.
//! CUDA kernels for yield simulation, thermal analysis, fault injection,
//! timing verification, weight compilation, and DRC.

pub mod tiler;
pub mod thermal;
pub mod fault;
pub mod verify;
pub mod compiler;
pub mod drc;

/// Vessel class specifications (28nm INT4 baseline)
#[derive(Debug, Clone, Copy)]
pub struct VesselSpec {
    pub name: &'static str,
    pub params_b: u64,
    pub power_w: f64,
    pub speed_toks: u32,
    pub die_mm2: f64,
}

pub const SCOUT: VesselSpec = VesselSpec { name: "Scout", params_b: 1_000_000_000, power_w: 0.8, speed_toks: 100, die_mm2: 25.0 };
pub const MESSENGER: VesselSpec = VesselSpec { name: "Messenger", params_b: 3_000_000_000, power_w: 2.5, speed_toks: 80, die_mm2: 49.0 };
pub const NAVIGATOR: VesselSpec = VesselSpec { name: "Navigator", params_b: 7_000_000_000, power_w: 5.0, speed_toks: 50, die_mm2: 100.0 };
pub const CAPTAIN: VesselSpec = VesselSpec { name: "Captain", params_b: 13_000_000_000, power_w: 10.0, speed_toks: 30, die_mm2: 196.0 };

pub const ALL_VESSELS: &[VesselSpec] = &[SCOUT, MESSENGER, NAVIGATOR, CAPTAIN];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vessel_specs() {
        assert_eq!(ALL_VESSELS.len(), 4);
        assert_eq!(SCOUT.params_b, 1_000_000_000);
        assert_eq!(CAPTAIN.die_mm2, 196.0);
    }
}
