# cuda-intelligence — GPU-Accelerated Frozen Intelligence Toolchain

Rust + CUDA rebuild of the frozen-intelligence Python toolchain. Optimized for real-time mask-locked inference chip design, verification, and yield simulation on NVIDIA GPUs.

## Architecture

```
cuda-intelligence/
├── Cargo.toml
├── src/
│   ├── lib.rs          — crate root, re-exports
│   ├── tiler.rs        — GPU yield-aware MoE swarm tiling
│   ├── thermal.rs      — GPU thermal simulation (finite-difference)
│   ├── fault.rs        — GPU fault simulation (stuck-at, bridging)
│   ├── verify.rs       — GPU timing/power/signoff verification
│   ├── compiler.rs     — GPU weight-to-metal compilation
│   └── drc.rs          — GPU design rule checking
└── README.md
```

## Vessel Classes

| Class | Params | Power | Speed | Die |
|-------|--------|-------|-------|-----|
| Scout | 1B | <1W | 100 tok/s | 25mm² |
| Messenger | 3B | 2.5W | 80 tok/s | 49mm² |
| Navigator | 7B | 5W | 50 tok/s | 100mm² |
| Captain | 13B | 10W | 30 tok/s | 196mm² |

## CUDA Patterns (borrowed from cudaclaw)

- **Persistent kernels** — <5μs dispatch overhead
- **Cell agents** — `repr(C)` structs on GPU
- **Muscle fibers** — SIMD-parallel compute paths
- **Ramify engine** — branch divergence management
- **SmartCRDT** — atomicCAS for concurrent state

## Quick Start

```bash
cargo build --release
cargo test
```

## Status

- [x] tiler.rs — Monte Carlo yield simulation on GPU
- [x] thermal.rs — 2D heat diffusion with thermal vias
- [x] fault.rs — stuck-at/bridging fault injection
- [x] verify.rs — timing analysis + power estimation
- [x] compiler.rs — weight quantization + METL binary format
- [x] drc.rs — parallel design rule checking
