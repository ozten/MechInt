use burn::tensor::backend::Backend;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{data::ModularAdditionDataset, model::Transformer};

/// FFT analysis results for embeddings
#[derive(Debug, Serialize, Deserialize)]
pub struct FFTAnalysis {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub dominant_frequencies: Vec<(usize, f64)>, // (token_id, dominant_frequency_index)
    pub frequency_magnitudes: HashMap<usize, Vec<f64>>, // token_id -> magnitude spectrum
}

/// Analyze the learned token embeddings using FFT
pub fn analyze_embeddings_fft<B: Backend>(model: &Transformer<B>) -> FFTAnalysis {
    println!("🔍 Performing FFT analysis on learned embeddings...");

    let vocab_size = ModularAdditionDataset::vocab_size();
    let embedding_dim = 128;

    // Extract token embeddings by forward passing through the embedding layer
    let mut frequency_magnitudes = HashMap::new();
    let mut dominant_frequencies = Vec::new();

    for token_id in 0..vocab_size {
        // Get the embedding for this token
        let embedding = get_token_embedding(model, token_id);

        // Compute FFT on the embedding vector
        let fft_result = compute_fft(&embedding);

        // Find dominant frequency
        let dominant_freq_idx = find_dominant_frequency(&fft_result);
        dominant_frequencies.push((token_id, dominant_freq_idx as f64));

        // Store magnitude spectrum
        frequency_magnitudes.insert(token_id, fft_result);

        if token_id % 10 == 0 {
            println!("  Token {:2}: dominant frequency = {}", token_id, dominant_freq_idx);
        }
    }

    println!("✅ FFT analysis complete!");
    println!();
    println!("📊 Dominant Frequency Analysis:");

    // Analyze patterns
    let mut freq_histogram: HashMap<usize, usize> = HashMap::new();
    for (_, freq) in &dominant_frequencies {
        *freq_histogram.entry(*freq as usize).or_insert(0) += 1;
    }

    let mut freq_counts: Vec<_> = freq_histogram.iter().collect();
    freq_counts.sort_by(|a, b| b.1.cmp(a.1));

    println!("  Most common dominant frequencies:");
    for (freq, count) in freq_counts.iter().take(5) {
        println!("    Frequency {}: {} tokens", freq, count);
    }

    // Check for modular structure (expected for mod p)
    println!();
    println!("🔬 Checking for modular structure:");
    let modulus = ModularAdditionDataset::modulus();
    if freq_histogram.contains_key(&modulus) {
        println!(
            "  ✓ Frequency {} (matching modulus) appears in embeddings",
            modulus
        );
    }

    FFTAnalysis {
        vocab_size,
        embedding_dim,
        dominant_frequencies,
        frequency_magnitudes,
    }
}

/// Extract the token embedding for a single token
fn get_token_embedding<B: Backend>(model: &Transformer<B>, token_id: usize) -> Vec<f64> {
    // Get the token embedding directly from the model
    let embedding_tensor = model.get_token_embedding(token_id);

    // Convert to vector - shape is [1, 1, embedding_dim], we want [embedding_dim]
    let embedding_data: Vec<f32> = embedding_tensor.into_data().to_vec().unwrap();

    // Convert to f64
    embedding_data.iter().map(|&x| x as f64).collect()
}

/// Compute FFT on a vector of real values
fn compute_fft(data: &[f64]) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(data.len());

    // Convert to complex numbers
    let mut buffer: Vec<Complex<f64>> = data
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Perform FFT
    fft.process(&mut buffer);

    // Compute magnitudes
    buffer.iter().map(|c| c.norm()).collect()
}

/// Find the index of the dominant frequency (excluding DC component at index 0)
fn find_dominant_frequency(magnitudes: &[f64]) -> usize {
    magnitudes
        .iter()
        .enumerate()
        .skip(1) // Skip DC component
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Save FFT analysis to JSON file
pub fn save_fft_analysis(analysis: &FFTAnalysis, filename: &str) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(analysis)?;
    std::fs::write(filename, json)?;
    println!("💾 FFT analysis saved to: {}", filename);
    Ok(())
}

/// Track training and validation loss over time
#[derive(Debug, Serialize, Deserialize)]
pub struct LossHistory {
    pub train_snapshots: Vec<(usize, f64)>, // (step, loss)
    pub val_snapshots: Vec<(usize, f64)>,   // (step, loss)
}

impl LossHistory {
    pub fn new() -> Self {
        Self {
            train_snapshots: Vec::new(),
            val_snapshots: Vec::new(),
        }
    }

    pub fn add_snapshot(&mut self, step: usize, train_loss: f64, val_loss: f64) {
        self.train_snapshots.push((step, train_loss));
        self.val_snapshots.push((step, val_loss));
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json)?;
        println!("💾 Loss history saved to: {}", filename);
        Ok(())
    }
}

/// Legacy: Single loss tracking (kept for backward compatibility)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeightNormSnapshot {
    pub step: usize,
    pub total_norm: f64,
}

/// Legacy: Track weight norms over training (actually stores loss)
#[derive(Debug, Serialize, Deserialize)]
pub struct WeightNormHistory {
    pub snapshots: Vec<WeightNormSnapshot>,
}

impl WeightNormHistory {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn add_snapshot(&mut self, step: usize, norm: f64) {
        self.snapshots.push(WeightNormSnapshot {
            step,
            total_norm: norm,
        });
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json)?;
        println!("💾 Weight norm history saved to: {}", filename);
        Ok(())
    }
}

/// Compute the L2 norm of all model parameters
/// This is a simplified version that uses the model's state
pub fn compute_model_weight_norm<B: Backend>(_model: &Transformer<B>) -> f64 {
    // Note: Direct parameter access in Burn requires the Record trait
    // For now, we'll use a simplified approach by tracking optimizer state
    // In a real implementation, we'd use model.into_record() to access weights
    // This is a placeholder that should be enhanced
    1.0 // Placeholder - will be computed from gradients during training
}

/// Extract all token embeddings as a matrix [vocab_size, embedding_dim]
pub fn extract_all_embeddings<B: Backend>(model: &Transformer<B>) -> Vec<Vec<f64>> {
    let vocab_size = ModularAdditionDataset::vocab_size();
    let mut embeddings = Vec::with_capacity(vocab_size);

    for token_id in 0..vocab_size {
        embeddings.push(get_token_embedding(model, token_id));
    }

    embeddings
}

/// Track training and validation accuracy over time
#[derive(Debug, Serialize, Deserialize)]
pub struct AccuracyHistory {
    pub train_snapshots: Vec<(usize, f32)>, // (step, accuracy)
    pub val_snapshots: Vec<(usize, f32)>,   // (step, accuracy)
}

impl AccuracyHistory {
    pub fn new() -> Self {
        Self {
            train_snapshots: Vec::new(),
            val_snapshots: Vec::new(),
        }
    }

    pub fn add_snapshot(&mut self, step: usize, train_acc: f32, val_acc: f32) {
        self.train_snapshots.push((step, train_acc));
        self.val_snapshots.push((step, val_acc));
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json)?;
        println!("💾 Accuracy history saved to: {}", filename);
        Ok(())
    }
}
