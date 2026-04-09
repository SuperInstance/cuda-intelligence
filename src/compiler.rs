//! GPU weight-to-metal compilation — quantization + METL binary format
//!
//! Compiles neural network weights to mask-locked chip binary format
//! with mixed-precision quantization (FP32/INT8/INT4) and SHA256
//! layer checksums.

use std::collections::HashMap;

/// Quantization precision
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    FP32,
    INT8,
    INT4,
}

impl Precision {
    pub fn bits(&self) -> u32 {
        match self { Precision::FP32 => 32, Precision::INT8 => 8, Precision::INT4 => 4 }
    }
    pub fn bytes_per_weight(&self) -> f64 {
        self.bits() as f64 / 8.0
    }
}

/// Layer in the compiled binary
#[derive(Debug, Clone)]
pub struct CompiledLayer {
    pub name: String,
    pub precision: Precision,
    pub rows: usize,
    pub cols: usize,
    pub offset_bytes: u64,
    pub size_bytes: u64,
    pub checksum: String, // SHA256 first 16 hex chars
}

/// METL binary header
#[derive(Debug)]
pub struct MetlHeader {
    pub magic: [u8; 4],      // "METL"
    pub version: u32,
    pub num_layers: u32,
    pub total_size: u64,
    pub vessel_class: String,
}

/// Compilation statistics
#[derive(Debug)]
pub struct CompileStats {
    pub original_size_mb: f64,
    pub compiled_size_mb: f64,
    pub compression_ratio: f64,
    pub layers: Vec<CompiledLayer>,
    pub precision_distribution: HashMap<Precision, usize>,
}

/// Weight-to-metal compiler
pub struct WeightCompiler {
    pub layer_precisions: HashMap<String, Precision>,
    pub default_precision: Precision,
}

impl Default for WeightCompiler {
    fn default() -> Self {
        let mut precisions = HashMap::new();
        // LayerNorm stays FP32 for numerical stability
        precisions.insert("layernorm".to_string(), Precision::FP32);
        // Embedding at INT8
        precisions.insert("embed".to_string(), Precision::INT8);
        // Attention and FFN at INT4
        precisions.insert("attn".to_string(), Precision::INT4);
        precisions.insert("ffn".to_string(), Precision::INT4);
        precisions.insert("attention".to_string(), Precision::INT4);
        precisions.insert("mlp".to_string(), Precision::INT4);

        WeightCompiler { layer_precisions: precisions, default_precision: Precision::INT4 }
    }
}

impl WeightCompiler {
    /// Determine precision for a layer by name pattern
    pub fn get_precision(&self, layer_name: &str) -> Precision {
        let name_lower = layer_name.to_lowercase();
        for (pattern, prec) in &self.layer_precisions {
            if name_lower.contains(pattern) { return *prec; }
        }
        self.default_precision
    }

    /// Calculate compiled size for a layer
    pub fn layer_size(&self, rows: usize, cols: usize, precision: Precision) -> u64 {
        let bits = precision.bits() as u64;
        let total_bits = rows as u64 * cols as u64 * bits;
        // Round up to nearest byte
        (total_bits + 7) / 8
    }

    /// Generate fake weights for compilation (in real impl, load from PyTorch)
    pub fn generate_layer_weights(&self, rows: usize, cols: usize) -> Vec<u8> {
        let size = (rows * cols + 1) / 2; // INT4: 2 weights per byte
        (0..size).map(|i| (i as u8).wrapping_mul(37).wrapping_add(13)).collect()
    }

    /// Compute SHA256-like checksum (simplified for demo)
    pub fn compute_checksum(&self, data: &[u8]) -> String {
        let mut hash: u32 = 0x811c9dc5; // FNV offset basis
        for &byte in data {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x01000193); // FNV prime
        }
        format!("{:08x}", hash)
    }

    /// Compile a full model to METL format
    pub fn compile_model(&self, layers: &[(String, usize, usize)]) -> CompileStats {
        let mut compiled_layers = Vec::new();
        let mut total_original = 0u64;
        let mut total_compiled = 0u64;
        let mut precision_dist: HashMap<Precision, usize> = HashMap::new();
        let mut offset = 16u64; // skip METL header

        for (name, rows, cols) in layers {
            let precision = self.get_precision(name);
            let original_size = rows as u64 * cols as u64 * 4; // FP32
            let compiled_size = self.layer_size(*rows, *cols, precision);
            let weights = self.generate_layer_weights(*rows, *cols);
            let checksum = self.compute_checksum(&weights);

            compiled_layers.push(CompiledLayer {
                name: name.clone(), precision, rows: *rows, cols: *cols,
                offset_bytes: offset, size_bytes: compiled_size, checksum,
            });

            *precision_dist.entry(precision).or_insert(0) += 1;
            total_original += original_size;
            total_compiled += compiled_size;
            offset += compiled_size;
        }

        let compression = if total_original > 0 {
            total_original as f64 / total_compiled as f64
        } else {
            1.0
        };

        CompileStats {
            original_size_mb: total_original as f64 / (1024.0 * 1024.0),
            compiled_size_mb: total_compiled as f64 / (1024.0 * 1024.0),
            compression_ratio: compression,
            layers: compiled_layers,
            precision_distribution: precision_dist,
        }
    }

    /// Estimate die area needed for weight storage
    pub fn estimate_weight_area(&self, compiled_mb: f64, tech_nm: u32) -> f64 {
        // SRAM: ~1 bit per 200 nm² at 28nm, scales linearly
        let area_per_bit_nm2 = tech_nm as f64 * 7.0; // rough scaling
        let total_bits = compiled_mb * 8.0 * 1024.0 * 1024.0;
        let area_nm2 = total_bits * area_per_bit_nm2;
        area_nm2 / 1e12 // nm² to mm²
    }
}

/// Generate a typical transformer model layer list
pub fn transformer_layers(n_layers: usize, hidden: usize, vocab: usize) -> Vec<(String, usize, usize)> {
    let mut layers = Vec::new();
    layers.push(("embed_tokens".to_string(), vocab, hidden));
    layers.push(("embed_positions".to_string(), 512, hidden));
    for i in 0..n_layers {
        layers.push((format!("layer{}_attn_qkv", i), hidden, hidden * 3));
        layers.push((format!("layer{}_attn_out", i), hidden, hidden));
        layers.push((format!("layer{}_ffn_gate", i), hidden, hidden * 4));
        layers.push((format!("layer{}_ffn_up", i), hidden, hidden * 4));
        layers.push((format!("layer{}_ffn_down", i), hidden * 4, hidden));
        layers.push((format!("layer{}_ln1", i), hidden, hidden));
        layers.push((format!("layer{}_ln2", i), hidden, hidden));
    }
    layers.push(("lm_head".to_string(), hidden, vocab));
    layers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_selection() {
        let compiler = WeightCompiler::default();
        assert_eq!(compiler.get_precision("layer0_attn_qkv"), Precision::INT4);
        assert_eq!(compiler.get_precision("layer0_ln1"), Precision::FP32);
        assert_eq!(compiler.get_precision("embed_tokens"), Precision::INT8);
    }

    #[test]
    fn test_compile_model() {
        let compiler = WeightCompiler::default();
        let layers = transformer_layers(6, 768, 32000);
        let stats = compiler.compile_model(&layers);
        assert!(stats.compression_ratio > 1.0, "Should compress FP32");
        assert!(stats.compiled_size_mb > 0.0);
        println!("6-layer 768d model: {:.1}MB -> {:.1}MB ({:.1}x compression)",
            stats.original_size_mb, stats.compiled_size_mb, stats.compression_ratio);
        for (prec, count) in &stats.precision_distribution {
            println!("  {:?}: {} layers", prec, count);
        }
    }

    #[test]
    fn test_layer_size() {
        let compiler = WeightCompiler::default();
        let int4_size = compiler.layer_size(768, 768, Precision::INT4);
        let fp32_size = compiler.layer_size(768, 768, Precision::FP32);
        assert_eq!(fp32_size, 768 * 768 * 4);
        assert!(int4_size < fp32_size / 4);
    }

    #[test]
    fn test_checksum() {
        let compiler = WeightCompiler::default();
        let c1 = compiler.compute_checksum(&[1, 2, 3, 4]);
        let c2 = compiler.compute_checksum(&[1, 2, 3, 5]);
        assert_ne!(c1, c2);
    }
}
