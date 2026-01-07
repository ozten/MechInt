use burn::{
    data::dataset::Dataset,
    nn::loss::CrossEntropyLossConfig,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

fn inverse_fft(data: &mut [Complex<f64>]) {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(data.len());
    fft.process(data);
}

fn top_k_frequency_indices(magnitudes: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = magnitudes.iter().copied().enumerate().collect();
    indexed.sort_by(|(idx_a, mag_a), (idx_b, mag_b)| {
        mag_b
            .partial_cmp(mag_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| idx_a.cmp(idx_b))
    });

    let keep = k.min(indexed.len());
    indexed.iter().take(keep).map(|(idx, _)| *idx).collect()
}

fn restrict_signal_to_top_k_frequencies(signal: &[f64], top_k: usize) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }

    if top_k == 0 {
        return vec![0.0; signal.len()];
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);

    let magnitudes: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();
    let keep = top_k_frequency_indices(&magnitudes, top_k);
    let keep_set: HashSet<usize> = keep.into_iter().collect();

    for (idx, value) in buffer.iter_mut().enumerate() {
        if !keep_set.contains(&idx) {
            *value = Complex::new(0.0, 0.0);
        }
    }

    inverse_fft(&mut buffer);

    let scale = 1.0 / signal.len() as f64;
    buffer.iter().map(|c| c.re * scale).collect()
}

fn exclude_top_k_frequencies(signal: &[f64], top_k: usize) -> Vec<f64> {
    if signal.is_empty() {
        return Vec::new();
    }

    if top_k == 0 {
        return signal.to_vec();
    }

    if top_k >= signal.len() {
        return vec![0.0; signal.len()];
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);

    let magnitudes: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();
    let remove = top_k_frequency_indices(&magnitudes, top_k);
    let remove_set: HashSet<usize> = remove.into_iter().collect();

    for (idx, value) in buffer.iter_mut().enumerate() {
        if remove_set.contains(&idx) {
            *value = Complex::new(0.0, 0.0);
        }
    }

    inverse_fft(&mut buffer);

    let scale = 1.0 / signal.len() as f64;
    buffer.iter().map(|c| c.re * scale).collect()
}

fn restricted_token_embedding_weights<B: Backend>(
    model: &Transformer<B>,
    top_k: usize,
) -> Vec<f32> {
    let weight = model.token_embedding_weights();
    let [vocab_size, d_model] = weight.dims();
    let weight_data: Vec<f32> = weight.into_data().to_vec().unwrap();
    let mut restricted = vec![0.0f32; weight_data.len()];

    for dim in 0..d_model {
        let mut column = Vec::with_capacity(vocab_size);
        for token in 0..vocab_size {
            column.push(weight_data[token * d_model + dim] as f64);
        }
        let filtered = restrict_signal_to_top_k_frequencies(&column, top_k);
        for token in 0..vocab_size {
            restricted[token * d_model + dim] = filtered[token] as f32;
        }
    }

    restricted
}

fn excluded_token_embedding_weights<B: Backend>(
    model: &Transformer<B>,
    top_k: usize,
) -> Vec<f32> {
    let weight = model.token_embedding_weights();
    let [vocab_size, d_model] = weight.dims();
    let weight_data: Vec<f32> = weight.into_data().to_vec().unwrap();
    let mut excluded = vec![0.0f32; weight_data.len()];

    for dim in 0..d_model {
        let mut column = Vec::with_capacity(vocab_size);
        for token in 0..vocab_size {
            column.push(weight_data[token * d_model + dim] as f64);
        }
        let filtered = exclude_top_k_frequencies(&column, top_k);
        for token in 0..vocab_size {
            excluded[token * d_model + dim] = filtered[token] as f32;
        }
    }

    excluded
}

pub fn compute_restricted_loss<B: Backend>(
    model: &Transformer<B>,
    dataset: &ModularAdditionDataset,
    device: &B::Device,
    top_k: usize,
    batch_size: usize,
) -> Result<f64, String> {
    if batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }

    let restricted_weights = restricted_token_embedding_weights(model, top_k);
    let [vocab_size, d_model] = model.token_embedding_weights().dims();
    let token_weights = Tensor::<B, 2>::from_data(
        TensorData::new(restricted_weights, [vocab_size, d_model]),
        device,
    );

    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut total_loss = 0.0f64;
    let mut total_batches = 0usize;

    for batch_start in (0..dataset.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset.len());
        let batch_len = batch_end - batch_start;
        if batch_len == 0 {
            break;
        }

        let mut inputs_vec = Vec::with_capacity(batch_len * 3);
        let mut targets_vec = Vec::with_capacity(batch_len);

        for idx in batch_start..batch_end {
            let (input, target) = dataset.get(idx).ok_or_else(|| {
                format!("missing dataset sample at index {}", idx)
            })?;
            inputs_vec.extend(input.iter().map(|value| *value as i32));
            targets_vec.push(target as i32);
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
            .reshape([batch_len, 3]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

        let logits = model.forward_with_token_weights(inputs, token_weights.clone());
        let loss = loss_fn.forward(logits, targets);
        let loss_value: f64 = loss.into_data().to_vec().unwrap().get(0).copied().unwrap_or(0.0);
        total_loss += loss_value;
        total_batches += 1;
    }

    if total_batches == 0 {
        return Err("no batches computed for restricted loss".to_string());
    }

    Ok(total_loss / total_batches as f64)
}

pub fn compute_excluded_loss<B: Backend>(
    model: &Transformer<B>,
    dataset: &ModularAdditionDataset,
    device: &B::Device,
    top_k: usize,
    batch_size: usize,
) -> Result<f64, String> {
    if batch_size == 0 {
        return Err("batch_size must be greater than 0".to_string());
    }

    let excluded_weights = excluded_token_embedding_weights(model, top_k);
    let [vocab_size, d_model] = model.token_embedding_weights().dims();
    let token_weights = Tensor::<B, 2>::from_data(
        TensorData::new(excluded_weights, [vocab_size, d_model]),
        device,
    );

    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut total_loss = 0.0f64;
    let mut total_batches = 0usize;

    for batch_start in (0..dataset.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset.len());
        let batch_len = batch_end - batch_start;
        if batch_len == 0 {
            break;
        }

        let mut inputs_vec = Vec::with_capacity(batch_len * 3);
        let mut targets_vec = Vec::with_capacity(batch_len);

        for idx in batch_start..batch_end {
            let (input, target) = dataset.get(idx).ok_or_else(|| {
                format!("missing dataset sample at index {}", idx)
            })?;
            inputs_vec.extend(input.iter().map(|value| *value as i32));
            targets_vec.push(target as i32);
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
            .reshape([batch_len, 3]);
        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

        let logits = model.forward_with_token_weights(inputs, token_weights.clone());
        let loss = loss_fn.forward(logits, targets);
        let loss_value: f64 = loss.into_data().to_vec().unwrap().get(0).copied().unwrap_or(0.0);
        total_loss += loss_value;
        total_batches += 1;
    }

    if total_batches == 0 {
        return Err("no batches computed for excluded loss".to_string());
    }

    Ok(total_loss / total_batches as f64)
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

/// Track excluded loss over time (top-k frequencies removed)
#[derive(Debug, Serialize, Deserialize)]
pub struct ExcludedLossHistory {
    pub snapshots: Vec<(usize, f64)>, // (step, loss)
}

impl ExcludedLossHistory {
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
        }
    }

    pub fn add_snapshot(&mut self, step: usize, loss: f64) {
        self.snapshots.push((step, loss));
    }

    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(filename, json)?;
        println!("💾 Excluded loss history saved to: {}", filename);
        Ok(())
    }
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
/// This returns the Frobenius norm (sum of squared weights) across all parameters
/// Note: This is a simplified implementation that computes the norm of token embeddings,
/// which contain the majority of the model parameters. For a complete implementation,
/// we would need to access all Linear layer weights via the Record trait.
pub fn compute_model_weight_norm<B: Backend>(model: &Transformer<B>) -> f64 {
    let mut total_norm_squared = 0.0f64;

    // Token embedding weights: [vocab_size, d_model]
    let token_emb_weights = model.token_embedding_weights();
    let token_emb_data: Vec<f32> = token_emb_weights.into_data().to_vec().unwrap();
    for &weight in &token_emb_data {
        total_norm_squared += (weight as f64) * (weight as f64);
    }

    // Note: Position embeddings, attention weights, MLP weights, and LM head weights
    // are not included in this simplified implementation. For weight decay verification,
    // tracking the token embeddings (which are the largest weight matrices) is sufficient
    // to observe the monotonic decrease under weight decay.

    total_norm_squared.sqrt()
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

/// Verify that FFT analysis of embeddings shows dominant frequencies
/// that align with the modulus structure and nontrivial spectral patterns
pub fn verify_fft_dominant_frequencies<B: Backend>(
    model: &Transformer<B>,
    config: &FFTDominantFrequencyConfig,
) -> Result<FFTDominantFrequencyReport, String> {
    let embeddings = extract_all_embeddings(model);
    verify_fft_dominant_frequencies_from_embeddings(&embeddings, config)
}

/// Verify FFT dominant frequencies from pre-extracted embeddings
pub fn verify_fft_dominant_frequencies_from_embeddings(
    embeddings: &[Vec<f64>],
    config: &FFTDominantFrequencyConfig,
) -> Result<FFTDominantFrequencyReport, String> {
    if embeddings.is_empty() {
        return Err("no embeddings provided".to_string());
    }

    let vocab_size = embeddings.len();
    if vocab_size < config.modulus {
        return Err(format!(
            "embedding count {} less than modulus {}",
            vocab_size, config.modulus
        ));
    }

    // Analyze each embedding dimension and collect dominant frequencies
    let mut frequency_histogram: HashMap<usize, usize> = HashMap::new();
    let embedding_dim = embeddings[0].len();

    // For each embedding dimension, compute FFT across all tokens and find dominant frequency
    for dim in 0..embedding_dim {
        let mut signal = Vec::with_capacity(vocab_size);
        for embedding in embeddings.iter().take(vocab_size) {
            if dim >= embedding.len() {
                return Err(format!(
                    "embedding dimension mismatch at dim {}",
                    dim
                ));
            }
            signal.push(embedding[dim]);
        }

        // Compute FFT and find dominant frequency
        let magnitudes = compute_fft(&signal);
        let dominant_freq = find_dominant_frequency(&magnitudes);
        *frequency_histogram.entry(dominant_freq).or_insert(0) += 1;
    }

    // Check 1: Modulus frequency should appear significantly
    let modulus_count = frequency_histogram.get(&config.modulus).copied().unwrap_or(0);
    let modulus_frequency_fraction = modulus_count as f64 / embedding_dim as f64;

    if modulus_frequency_fraction < config.min_modulus_frequency_fraction {
        return Err(format!(
            "modulus frequency {} appears in only {:.3} of dimensions (threshold {:.3})",
            config.modulus, modulus_frequency_fraction, config.min_modulus_frequency_fraction
        ));
    }

    // Check 2: Spectrum should NOT resemble white noise (low entropy)
    // High entropy = uniform distribution (white noise), low entropy = structured peaks
    let total_dims = embedding_dim as f64;
    let mut entropy = 0.0;
    for &count in frequency_histogram.values() {
        if count > 0 {
            let p = count as f64 / total_dims;
            entropy -= p * p.log2();
        }
    }
    // Normalize entropy by max possible (log2 of unique frequency count)
    let max_entropy = (frequency_histogram.len() as f64).log2();
    let normalized_entropy = if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    };

    if normalized_entropy > config.max_spectrum_entropy {
        return Err(format!(
            "spectrum entropy {:.3} too high (white noise threshold {:.3})",
            normalized_entropy, config.max_spectrum_entropy
        ));
    }

    // Check 3: Peak frequencies should be significantly stronger than median
    let mut counts: Vec<usize> = frequency_histogram.values().copied().collect();
    counts.sort_unstable();
    let median_count = if counts.is_empty() {
        0
    } else {
        counts[counts.len() / 2]
    };
    let peak_count = counts.last().copied().unwrap_or(0);
    let peak_to_median_ratio = if median_count > 0 {
        peak_count as f64 / median_count as f64
    } else {
        f64::INFINITY
    };

    if peak_to_median_ratio < config.min_peak_to_median_ratio {
        return Err(format!(
            "peak to median ratio {:.3} below threshold {:.3}",
            peak_to_median_ratio, config.min_peak_to_median_ratio
        ));
    }

    // Collect top frequencies for the report
    let mut freq_counts: Vec<(usize, usize)> = frequency_histogram.into_iter().collect();
    freq_counts.sort_by(|a, b| b.1.cmp(&a.1));
    let top_frequencies: Vec<(usize, usize)> = freq_counts.into_iter().take(5).collect();

    Ok(FFTDominantFrequencyReport {
        modulus_frequency_fraction,
        spectrum_entropy: normalized_entropy,
        peak_to_median_ratio,
        top_frequencies,
    })
}

#[derive(Debug, Clone)]
pub struct FFTDominantFrequencyConfig {
    pub modulus: usize,
    pub min_modulus_frequency_fraction: f64,
    pub max_spectrum_entropy: f64,
    pub min_peak_to_median_ratio: f64,
}

impl FFTDominantFrequencyConfig {
    pub fn default_for_modulus(modulus: usize) -> Self {
        Self {
            modulus,
            min_modulus_frequency_fraction: 0.15,
            max_spectrum_entropy: 0.85,
            min_peak_to_median_ratio: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FFTDominantFrequencyReport {
    pub modulus_frequency_fraction: f64,
    pub spectrum_entropy: f64,
    pub peak_to_median_ratio: f64,
    pub top_frequencies: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingClockConfig {
    pub max_radius_std_fraction: f64,
    pub min_angular_coverage: f64,
    pub max_mean_angle_error: f64,
    pub min_order_fraction: f64,
}

impl EmbeddingClockConfig {
    pub fn default_for_modulus(_modulus: usize) -> Self {
        Self {
            max_radius_std_fraction: 0.1,
            min_angular_coverage: 0.9,
            max_mean_angle_error: 0.35,
            min_order_fraction: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingClockReport {
    pub mean_radius: f64,
    pub radius_std: f64,
    pub angular_coverage: f64,
    pub mean_angle_error: f64,
    pub order_fraction: f64,
}

pub fn project_embeddings_to_fourier_plane(
    embeddings: &[Vec<f64>],
    modulus: usize,
) -> Result<Vec<(f64, f64)>, String> {
    let p = modulus;
    if embeddings.len() < p {
        return Err("embedding matrix smaller than modulus".to_string());
    }
    if p == 0 {
        return Err("modulus must be positive".to_string());
    }
    let d = embeddings[0].len();
    if d == 0 {
        return Err("embedding dimension must be positive".to_string());
    }
    if !embeddings[..p].iter().all(|row| row.len() == d) {
        return Err("embedding rows have inconsistent dimension".to_string());
    }

    let tau = std::f64::consts::PI * 2.0;
    let mut basis_cos = vec![0.0f64; d];
    let mut basis_sin = vec![0.0f64; d];
    let scale = 2.0 / p as f64;

    for (token, embedding) in embeddings[..p].iter().enumerate() {
        let angle = tau * token as f64 / p as f64;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        for dim in 0..d {
            basis_cos[dim] += cos_val * embedding[dim] * scale;
            basis_sin[dim] += sin_val * embedding[dim] * scale;
        }
    }

    let basis_cos = normalize_vector(&basis_cos)?;
    let basis_sin = normalize_vector(&basis_sin)?;

    let mut coords = Vec::with_capacity(p);
    for embedding in embeddings[..p].iter() {
        let x = dot(embedding, &basis_cos);
        let y = dot(embedding, &basis_sin);
        coords.push((x, y));
    }

    Ok(coords)
}

pub fn verify_embedding_clock_geometry(
    embeddings: &[Vec<f64>],
    modulus: usize,
    config: &EmbeddingClockConfig,
) -> Result<EmbeddingClockReport, String> {
    let coords = project_embeddings_to_fourier_plane(embeddings, modulus)?;
    let p = coords.len();
    if p == 0 {
        return Err("no embedding coordinates computed".to_string());
    }

    let mut radii = Vec::with_capacity(p);
    let mut angles = Vec::with_capacity(p);
    for (x, y) in &coords {
        let radius = (x * x + y * y).sqrt();
        radii.push(radius);
        angles.push(y.atan2(*x));
    }

    let mean_radius = radii.iter().sum::<f64>() / p as f64;
    if mean_radius == 0.0 {
        return Err("mean radius is zero".to_string());
    }
    let radius_var = radii
        .iter()
        .map(|r| (r - mean_radius) * (r - mean_radius))
        .sum::<f64>()
        / p as f64;
    let radius_std = radius_var.sqrt();

    if radius_std / mean_radius > config.max_radius_std_fraction {
        return Err(format!(
            "radius variance too large (std {:.4}, mean {:.4})",
            radius_std, mean_radius
        ));
    }

    let tau = std::f64::consts::PI * 2.0;
    let (mut sum_cos, mut sum_sin) = (0.0f64, 0.0f64);
    for (token, angle) in angles.iter().enumerate() {
        let expected = tau * token as f64 / p as f64;
        let delta = angle - expected;
        sum_cos += delta.cos();
        sum_sin += delta.sin();
    }
    let offset = sum_sin.atan2(sum_cos);

    let mut adjusted_angles = Vec::with_capacity(p);
    let mut angle_errors = Vec::with_capacity(p);
    for (token, angle) in angles.iter().enumerate() {
        let expected = tau * token as f64 / p as f64;
        let mut adjusted = angle - offset;
        adjusted = wrap_angle(adjusted);
        adjusted_angles.push(adjusted);
        let diff = wrap_angle_signed(adjusted - expected);
        angle_errors.push(diff.abs());
    }

    let mean_angle_error = angle_errors.iter().sum::<f64>() / p as f64;
    if mean_angle_error > config.max_mean_angle_error {
        return Err(format!(
            "mean angular error too large ({:.4} rad)",
            mean_angle_error
        ));
    }

    let angular_coverage = angular_coverage(&adjusted_angles);
    if angular_coverage < config.min_angular_coverage {
        return Err(format!(
            "angular coverage too small ({:.4})",
            angular_coverage
        ));
    }

    let order_fraction = order_fraction(&adjusted_angles, p);
    if order_fraction < config.min_order_fraction {
        return Err(format!(
            "token ordering too weak ({:.4})",
            order_fraction
        ));
    }

    Ok(EmbeddingClockReport {
        mean_radius,
        radius_std,
        angular_coverage,
        mean_angle_error,
        order_fraction,
    })
}

fn normalize_vector(vec: &[f64]) -> Result<Vec<f64>, String> {
    let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err("basis vector norm is zero".to_string());
    }
    Ok(vec.iter().map(|v| v / norm).collect())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn wrap_angle(angle: f64) -> f64 {
    let tau = std::f64::consts::PI * 2.0;
    let mut wrapped = angle % tau;
    if wrapped < 0.0 {
        wrapped += tau;
    }
    wrapped
}

fn wrap_angle_signed(angle: f64) -> f64 {
    let tau = std::f64::consts::PI * 2.0;
    let mut wrapped = (angle + std::f64::consts::PI) % tau;
    if wrapped < 0.0 {
        wrapped += tau;
    }
    wrapped - std::f64::consts::PI
}

fn angular_coverage(angles: &[f64]) -> f64 {
    if angles.is_empty() {
        return 0.0;
    }
    let tau = std::f64::consts::PI * 2.0;
    let mut sorted = angles.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_gap = 0.0;
    for pair in sorted.windows(2) {
        let gap = pair[1] - pair[0];
        if gap > max_gap {
            max_gap = gap;
        }
    }
    let wrap_gap = tau - (sorted[sorted.len() - 1] - sorted[0]);
    if wrap_gap > max_gap {
        max_gap = wrap_gap;
    }

    (tau - max_gap) / tau
}

fn order_fraction(angles: &[f64], modulus: usize) -> f64 {
    if angles.len() < 2 || modulus == 0 {
        return 0.0;
    }
    let mut indexed: Vec<(usize, f64)> = angles.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut good = 0usize;
    for window in indexed.windows(2) {
        let current = window[0].0;
        let next = window[1].0;
        if (next + modulus - current) % modulus == 1 {
            good += 1;
        }
    }
    let last = indexed[indexed.len() - 1].0;
    let first = indexed[0].0;
    if (first + modulus - last) % modulus == 1 {
        good += 1;
    }

    good as f64 / modulus as f64
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_circle_embeddings(modulus: usize) -> Vec<Vec<f64>> {
        let tau = std::f64::consts::PI * 2.0;
        (0..modulus)
            .map(|token| {
                let angle = tau * token as f64 / modulus as f64;
                vec![angle.cos(), angle.sin(), 0.0, 0.0]
            })
            .collect()
    }

    #[test]
    fn top_k_frequency_indices_stable_with_ties() {
        let magnitudes = vec![0.1, 1.0, 1.0, 0.5];
        let first = top_k_frequency_indices(&magnitudes, 2);
        let second = top_k_frequency_indices(&magnitudes, 2);
        assert_eq!(first, vec![1, 2]);
        assert_eq!(first, second);
    }

    #[test]
    fn restrict_signal_zeroes_when_top_k_zero() {
        let signal = vec![1.0, -1.0, 0.5, -0.5];
        let filtered = restrict_signal_to_top_k_frequencies(&signal, 0);
        assert_eq!(filtered, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn exclude_signal_no_change_when_top_k_zero() {
        let signal = vec![1.0, -1.0, 0.5, -0.5];
        let filtered = exclude_top_k_frequencies(&signal, 0);
        assert_eq!(filtered, signal);
    }

    #[test]
    fn exclude_signal_zeroes_when_top_k_all() {
        let signal = vec![1.0, -1.0, 0.5, -0.5];
        let filtered = exclude_top_k_frequencies(&signal, signal.len());
        assert_eq!(filtered, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn embedding_clock_geometry_passes_on_clean_circle() {
        let modulus = 13;
        let embeddings = make_circle_embeddings(modulus);
        let config = EmbeddingClockConfig::default_for_modulus(modulus);
        let report =
            verify_embedding_clock_geometry(&embeddings, modulus, &config).expect("expected pass");
        assert!(report.angular_coverage > 0.9);
        assert!(report.order_fraction > 0.9);
        assert!(report.mean_angle_error < 0.1);
    }

    #[test]
    fn embedding_clock_geometry_flags_wrong_frequency() {
        let modulus = 11;
        let tau = std::f64::consts::PI * 2.0;
        let embeddings: Vec<Vec<f64>> = (0..modulus)
            .map(|token| {
                let angle = tau * token as f64 / modulus as f64;
                vec![(2.0 * angle).cos(), (2.0 * angle).sin(), 0.0, 0.0]
            })
            .collect();
        let config = EmbeddingClockConfig {
            max_radius_std_fraction: 0.5,
            min_angular_coverage: 0.1,
            max_mean_angle_error: 0.2,
            min_order_fraction: 0.1,
        };
        let err = verify_embedding_clock_geometry(&embeddings, modulus, &config)
            .expect_err("expected ordering verification to fail");
        assert!(err.contains("mean angular error"));
    }

    #[test]
    fn fft_dominant_frequency_passes_on_modulus_structure() {
        // Create embeddings with strong modulus-113 frequency component
        // Mix in some other frequencies to get a realistic peak-to-median ratio
        let modulus = 113;
        let vocab_size = 114; // 0-112 + equals token
        let embedding_dim = 128;
        let tau = std::f64::consts::PI * 2.0;

        let mut embeddings = Vec::with_capacity(vocab_size);
        for token in 0..vocab_size {
            let mut embedding = Vec::with_capacity(embedding_dim);
            for dim in 0..embedding_dim {
                // 60% of dimensions have modulus frequency, 40% have other frequencies
                let freq = if dim % 5 < 3 {
                    modulus
                } else {
                    // Use various other frequencies for the remaining dims
                    20 + (dim % 20)
                };
                let angle = tau * freq as f64 * token as f64 / vocab_size as f64
                    + dim as f64 * 0.1;
                embedding.push(angle.sin());
            }
            embeddings.push(embedding);
        }

        let config = FFTDominantFrequencyConfig::default_for_modulus(modulus);
        let report = verify_fft_dominant_frequencies_from_embeddings(&embeddings, &config)
            .expect("expected modulus structure to pass");

        assert!(report.modulus_frequency_fraction >= config.min_modulus_frequency_fraction);
        assert!(report.spectrum_entropy <= config.max_spectrum_entropy);
        assert!(report.peak_to_median_ratio >= config.min_peak_to_median_ratio);
        assert!(report.top_frequencies.iter().any(|(freq, _)| *freq == modulus));
    }

    #[test]
    fn fft_dominant_frequency_fails_on_white_noise() {
        // Create random embeddings (white noise spectrum)
        let modulus = 113;
        let vocab_size = 114;
        let embedding_dim = 128;

        let mut embeddings = Vec::with_capacity(vocab_size);
        for token in 0..vocab_size {
            let mut embedding = Vec::with_capacity(embedding_dim);
            for dim in 0..embedding_dim {
                // Pseudo-random values
                let value = ((token * 17 + dim * 31) % 100) as f64 / 50.0 - 1.0;
                embedding.push(value);
            }
            embeddings.push(embedding);
        }

        let config = FFTDominantFrequencyConfig::default_for_modulus(modulus);
        let err = verify_fft_dominant_frequencies_from_embeddings(&embeddings, &config)
            .expect_err("expected white noise to fail");

        // Should fail on either modulus frequency fraction or spectrum entropy
        assert!(
            err.contains("modulus frequency") || err.contains("spectrum entropy")
                || err.contains("peak to median")
        );
    }

    #[test]
    fn fft_dominant_frequency_fails_on_wrong_frequency() {
        // Create embeddings with dominant frequency at 57 (not 113)
        let modulus = 113;
        let vocab_size = 114;
        let embedding_dim = 128;
        let tau = std::f64::consts::PI * 2.0;
        let wrong_frequency = 57;

        let mut embeddings = Vec::with_capacity(vocab_size);
        for token in 0..vocab_size {
            let mut embedding = Vec::with_capacity(embedding_dim);
            for dim in 0..embedding_dim {
                // Each dimension has a sine wave with wrong frequency
                let angle = tau * wrong_frequency as f64 * token as f64 / vocab_size as f64
                    + dim as f64 * 0.1;
                embedding.push(angle.sin());
            }
            embeddings.push(embedding);
        }

        let config = FFTDominantFrequencyConfig::default_for_modulus(modulus);
        let err = verify_fft_dominant_frequencies_from_embeddings(&embeddings, &config)
            .expect_err("expected wrong frequency to fail");

        assert!(err.contains("modulus frequency"));
    }

    #[test]
    fn fft_dominant_frequency_fails_on_flat_embeddings() {
        // Create constant embeddings (DC component only, no structure)
        let modulus = 113;
        let vocab_size = 114;
        let embedding_dim = 128;

        let embeddings = vec![vec![1.5; embedding_dim]; vocab_size];

        let config = FFTDominantFrequencyConfig::default_for_modulus(modulus);
        let err = verify_fft_dominant_frequencies_from_embeddings(&embeddings, &config)
            .expect_err("expected flat embeddings to fail");

        assert!(err.contains("modulus frequency") || err.contains("peak to median"));
    }

    #[test]
    fn fft_dominant_frequency_fails_on_weak_peaks() {
        // Create embeddings with many frequencies at similar magnitudes (no clear peaks)
        let modulus = 113;
        let vocab_size = 114;
        let embedding_dim = 128;
        let tau = std::f64::consts::PI * 2.0;

        let mut embeddings = Vec::with_capacity(vocab_size);
        for token in 0..vocab_size {
            let mut embedding = Vec::with_capacity(embedding_dim);
            for dim in 0..embedding_dim {
                // Mix many frequencies with similar amplitudes
                let mut value = 0.0;
                for freq in 1..10 {
                    let angle =
                        tau * freq as f64 * token as f64 / vocab_size as f64 + dim as f64 * 0.01;
                    value += angle.sin() / 10.0;
                }
                embedding.push(value);
            }
            embeddings.push(embedding);
        }

        let config = FFTDominantFrequencyConfig {
            modulus,
            min_modulus_frequency_fraction: 0.15,
            max_spectrum_entropy: 0.85,
            min_peak_to_median_ratio: 2.0,
        };
        let err = verify_fft_dominant_frequencies_from_embeddings(&embeddings, &config)
            .expect_err("expected weak peaks to fail");

        assert!(
            err.contains("modulus frequency") || err.contains("spectrum entropy")
                || err.contains("peak to median")
        );
    }
}
