use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Int, Tensor},
};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::analysis::{AccuracyHistory, LossHistory};
use crate::data::ModularAdditionDataset;
use crate::model::Transformer;

#[derive(Debug, Clone)]
pub struct GrokkingVerificationConfig {
    pub early_train_acc_threshold: f32,
    pub early_step_max: usize,
    pub plateau_min_step: usize,
    pub generalization_window: usize,
    pub chance_accuracy: f32,
    pub chance_tolerance: f32,
    pub target_val_acc_threshold: f32,
    pub loss_window: usize,
    pub loss_drop_fraction: f64,
}

impl GrokkingVerificationConfig {
    pub fn default_for_modulus(modulus: usize) -> Self {
        Self {
            early_train_acc_threshold: 0.99,
            early_step_max: 1000,
            plateau_min_step: 1000,
            generalization_window: 10_000,
            chance_accuracy: 1.0 / modulus as f32,
            chance_tolerance: 0.02,
            target_val_acc_threshold: 0.99,
            loss_window: 5,
            loss_drop_fraction: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrokkingPhaseReport {
    pub early_train_step: usize,
    pub transition_step: usize,
    pub plateau_max_val_acc: f32,
    pub loss_drop_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct RestrictedLossVerificationConfig {
    pub plateau_min_step: usize,
    pub drop_window: usize,
    pub drop_fraction: f64,
    pub min_step_lead: usize,
}

#[derive(Debug, Clone)]
pub struct RestrictedLossReport {
    pub restricted_drop_step: usize,
    pub full_drop_step: usize,
}

#[derive(Debug, Clone)]
pub struct ExcludedLossSpikeConfig {
    pub min_relative_increase: f64,
}

#[derive(Debug, Clone)]
pub struct ExcludedLossSpikeReport {
    pub spike_step: usize,
    pub spike_loss: f64,
    pub baseline_loss: f64,
}

#[derive(Debug, Clone)]
pub struct MlpWaveVerificationConfig {
    pub max_negative_fraction: f64,
    pub negative_tolerance: f64,
    pub min_dominant_ratio: f64,
    pub min_correlation: f64,
    pub phase_steps: usize,
}

#[derive(Debug, Clone)]
pub struct MlpWaveNeuronReport {
    pub neuron_index: usize,
    pub dominant_frequency: usize,
    pub dominant_ratio: f64,
    pub negative_fraction: f64,
    pub best_correlation: f64,
}

#[derive(Debug, Clone)]
pub struct MlpWaveReport {
    pub neuron_reports: Vec<MlpWaveNeuronReport>,
}

pub fn verify_grokking_phase_transition(
    loss_history: &LossHistory,
    accuracy_history: &AccuracyHistory,
    config: &GrokkingVerificationConfig,
) -> Result<GrokkingPhaseReport, String> {
    let (early_train_step, _) = accuracy_history
        .train_snapshots
        .iter()
        .find(|(_, acc)| *acc >= config.early_train_acc_threshold)
        .ok_or_else(|| "train accuracy never reaches threshold".to_string())?;

    if *early_train_step > config.early_step_max {
        return Err(format!(
            "train accuracy reaches {:.2} after step {}, expected before {}",
            config.early_train_acc_threshold, early_train_step, config.early_step_max
        ));
    }

    let plateau_max_val_acc = accuracy_history
        .val_snapshots
        .iter()
        .filter(|(step, _)| *step <= config.plateau_min_step)
        .map(|(_, acc)| *acc)
        .fold(None::<f32>, |accum, value| {
            Some(accum.map_or(value, |max_val| max_val.max(value)))
        })
        .ok_or_else(|| "no validation accuracy samples before plateau_min_step".to_string())?;

    if plateau_max_val_acc > config.chance_accuracy + config.chance_tolerance {
        return Err(format!(
            "validation accuracy before step {} exceeds chance (max {:.4})",
            config.plateau_min_step, plateau_max_val_acc
        ));
    }

    let (transition_step, _) = accuracy_history
        .val_snapshots
        .iter()
        .find(|(step, acc)| {
            *step >= config.plateau_min_step && *acc >= config.target_val_acc_threshold
        })
        .ok_or_else(|| "validation accuracy never reaches target threshold".to_string())?;

    let latest_allowed = config.plateau_min_step + config.generalization_window;
    if *transition_step > latest_allowed {
        return Err(format!(
            "validation accuracy reaches {:.2} at step {}, expected before {}",
            config.target_val_acc_threshold, transition_step, latest_allowed
        ));
    }

    let loss_drop_ratio = verify_loss_drop(
        &loss_history.val_snapshots,
        *transition_step,
        config.loss_window,
        config.loss_drop_fraction,
    )?;

    Ok(GrokkingPhaseReport {
        early_train_step: *early_train_step,
        transition_step: *transition_step,
        plateau_max_val_acc,
        loss_drop_ratio,
    })
}

fn verify_loss_drop(
    val_losses: &[(usize, f64)],
    transition_step: usize,
    loss_window: usize,
    loss_drop_fraction: f64,
) -> Result<f64, String> {
    let before_losses: Vec<f64> = val_losses
        .iter()
        .filter(|(step, _)| *step < transition_step)
        .map(|(_, loss)| *loss)
        .collect();

    let after_losses: Vec<f64> = val_losses
        .iter()
        .filter(|(step, _)| *step >= transition_step)
        .map(|(_, loss)| *loss)
        .collect();

    if before_losses.len() < loss_window || after_losses.len() < loss_window {
        return Err("insufficient loss samples around transition".to_string());
    }

    let before_avg: f64 =
        before_losses[before_losses.len() - loss_window..].iter().sum::<f64>()
            / loss_window as f64;
    let after_avg: f64 = after_losses[..loss_window].iter().sum::<f64>() / loss_window as f64;

    if after_avg > before_avg * (1.0 - loss_drop_fraction) {
        return Err(format!(
            "validation loss drop too small (before {:.4}, after {:.4})",
            before_avg, after_avg
        ));
    }

    Ok(after_avg / before_avg)
}

pub fn verify_restricted_loss_early_drop(
    full_losses: &[(usize, f64)],
    restricted_losses: &[(usize, f64)],
    config: &RestrictedLossVerificationConfig,
) -> Result<RestrictedLossReport, String> {
    let restricted_baseline =
        baseline_loss(restricted_losses, config.plateau_min_step, config.drop_window)?;
    let full_baseline = baseline_loss(full_losses, config.plateau_min_step, config.drop_window)?;

    let restricted_drop_step = find_drop_step(
        restricted_losses,
        config.plateau_min_step,
        restricted_baseline,
        config.drop_fraction,
    )?;
    let full_drop_step = find_drop_step(
        full_losses,
        config.plateau_min_step,
        full_baseline,
        config.drop_fraction,
    )?;

    if restricted_drop_step + config.min_step_lead > full_drop_step {
        return Err(format!(
            "restricted loss drops at step {}, expected at least {} steps before full loss (step {})",
            restricted_drop_step, config.min_step_lead, full_drop_step
        ));
    }

    Ok(RestrictedLossReport {
        restricted_drop_step,
        full_drop_step,
    })
}

pub fn verify_excluded_loss_spike(
    excluded_losses: &[(usize, f64)],
    config: &ExcludedLossSpikeConfig,
) -> Result<ExcludedLossSpikeReport, String> {
    if excluded_losses.len() < 3 {
        return Err("excluded loss history requires at least 3 points".to_string());
    }

    let first = excluded_losses.first().ok_or_else(|| {
        "excluded loss history is empty".to_string()
    })?;
    let last = excluded_losses.last().ok_or_else(|| {
        "excluded loss history is empty".to_string()
    })?;
    let baseline_loss = (first.1 + last.1) / 2.0;

    let (peak_idx, (peak_step, peak_loss)) = excluded_losses
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| "excluded loss history is empty".to_string())?;

    if peak_idx == 0 || peak_idx + 1 == excluded_losses.len() {
        return Err("excluded loss spike occurs at boundary".to_string());
    }

    if peak_loss < &(baseline_loss * (1.0 + config.min_relative_increase)) {
        return Err(format!(
            "excluded loss spike too small (baseline {:.4}, peak {:.4})",
            baseline_loss, peak_loss
        ));
    }

    Ok(ExcludedLossSpikeReport {
        spike_step: *peak_step,
        spike_loss: *peak_loss,
        baseline_loss,
    })
}

pub fn verify_mlp_activation_waves<B: Backend>(
    model: &Transformer<B>,
    device: &B::Device,
    fixed_y: usize,
    neuron_indices: &[usize],
    config: &MlpWaveVerificationConfig,
) -> Result<MlpWaveReport, String> {
    let activations = collect_mlp_post_relu_activations(model, device, fixed_y)?;
    if neuron_indices.is_empty() {
        return Err("neuron indices list is empty".to_string());
    }

    let d_ff = activations.len();
    for &idx in neuron_indices {
        if idx >= d_ff {
            return Err(format!(
                "neuron index {} out of range (d_ff={})",
                idx, d_ff
            ));
        }
    }

    let signals: Vec<(usize, Vec<f64>)> = neuron_indices
        .iter()
        .map(|&idx| (idx, activations[idx].clone()))
        .collect();
    verify_mlp_activation_waves_from_signals(&signals, config)
}

pub fn verify_mlp_activation_waves_from_signals(
    signals: &[(usize, Vec<f64>)],
    config: &MlpWaveVerificationConfig,
) -> Result<MlpWaveReport, String> {
    if signals.is_empty() {
        return Err("no activation signals provided".to_string());
    }

    let mut reports = Vec::with_capacity(signals.len());
    for (neuron_index, signal) in signals {
        let report = analyze_mlp_activation_wave(*neuron_index, signal, config)?;
        reports.push(report);
    }

    Ok(MlpWaveReport {
        neuron_reports: reports,
    })
}

fn baseline_loss(
    losses: &[(usize, f64)],
    plateau_min_step: usize,
    drop_window: usize,
) -> Result<f64, String> {
    let before: Vec<f64> = losses
        .iter()
        .filter(|(step, _)| *step <= plateau_min_step)
        .map(|(_, loss)| *loss)
        .collect();

    if before.len() < drop_window || drop_window == 0 {
        return Err("insufficient loss samples for baseline".to_string());
    }

    let window = &before[before.len() - drop_window..];
    Ok(window.iter().sum::<f64>() / drop_window as f64)
}

fn find_drop_step(
    losses: &[(usize, f64)],
    plateau_min_step: usize,
    baseline: f64,
    drop_fraction: f64,
) -> Result<usize, String> {
    losses
        .iter()
        .filter(|(step, _)| *step >= plateau_min_step)
        .find(|(_, loss)| *loss <= baseline * (1.0 - drop_fraction))
        .map(|(step, _)| *step)
        .ok_or_else(|| "loss never drops below threshold".to_string())
}

fn collect_mlp_post_relu_activations<B: Backend>(
    model: &Transformer<B>,
    device: &B::Device,
    fixed_y: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let modulus = ModularAdditionDataset::modulus();
    let equals_token = ModularAdditionDataset::equals_token();

    let mut inputs_vec = Vec::with_capacity(modulus * 3);
    for x in 0..modulus {
        inputs_vec.push(x as i32);
        inputs_vec.push(fixed_y as i32);
        inputs_vec.push(equals_token as i32);
    }

    let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
        .reshape([modulus, 3]);
    let (_, mlp_acts) = model.forward_with_mlp_activations(inputs);
    let [batch_size, d_ff] = mlp_acts.dims();

    if batch_size != modulus {
        return Err(format!(
            "unexpected activation batch size {} (expected {})",
            batch_size, modulus
        ));
    }

    let data: Vec<f32> = mlp_acts.into_data().to_vec().unwrap();
    let mut per_neuron = vec![Vec::with_capacity(batch_size); d_ff];
    for batch in 0..batch_size {
        for neuron in 0..d_ff {
            let idx = batch * d_ff + neuron;
            per_neuron[neuron].push(data[idx] as f64);
        }
    }

    Ok(per_neuron)
}

fn analyze_mlp_activation_wave(
    neuron_index: usize,
    signal: &[f64],
    config: &MlpWaveVerificationConfig,
) -> Result<MlpWaveNeuronReport, String> {
    if signal.len() < 3 {
        return Err(format!(
            "activation signal for neuron {} is too short",
            neuron_index
        ));
    }

    let negative_count = signal
        .iter()
        .filter(|value| **value < -config.negative_tolerance)
        .count();
    let negative_fraction = negative_count as f64 / signal.len() as f64;
    if negative_fraction > config.max_negative_fraction {
        return Err(format!(
            "neuron {} has {:.3} negative activations",
            neuron_index, negative_fraction
        ));
    }

    let (dominant_frequency, dominant_ratio) = dominant_frequency_ratio(signal)?;
    if dominant_ratio < config.min_dominant_ratio {
        return Err(format!(
            "neuron {} dominant frequency ratio {:.3} below threshold {:.3}",
            neuron_index, dominant_ratio, config.min_dominant_ratio
        ));
    }

    let best_correlation =
        best_rectified_sine_correlation(signal, dominant_frequency, config.phase_steps);
    if best_correlation < config.min_correlation {
        return Err(format!(
            "neuron {} rectified sine correlation {:.3} below threshold {:.3}",
            neuron_index, best_correlation, config.min_correlation
        ));
    }

    Ok(MlpWaveNeuronReport {
        neuron_index,
        dominant_frequency,
        dominant_ratio,
        negative_fraction,
        best_correlation,
    })
}

fn dominant_frequency_ratio(signal: &[f64]) -> Result<(usize, f64), String> {
    if signal.len() < 2 {
        return Err("signal too short for FFT".to_string());
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());
    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    fft.process(&mut buffer);
    let magnitudes: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();

    if magnitudes.len() < 2 {
        return Err("insufficient FFT bins".to_string());
    }

    let mut dominant_idx = 1usize;
    let mut dominant_mag = magnitudes[1];
    for (idx, &mag) in magnitudes.iter().enumerate().skip(2) {
        if mag > dominant_mag {
            dominant_mag = mag;
            dominant_idx = idx;
        }
    }

    if dominant_idx == 0 {
        return Err("dominant frequency is zero".to_string());
    }

    let mut nonzero = magnitudes[1..].to_vec();
    nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_mag = nonzero[nonzero.len() / 2];
    let ratio = if median_mag == 0.0 {
        dominant_mag
    } else {
        dominant_mag / median_mag
    };

    Ok((dominant_idx, ratio))
}

fn best_rectified_sine_correlation(signal: &[f64], frequency: usize, phase_steps: usize) -> f64 {
    if signal.is_empty() || frequency == 0 || phase_steps == 0 {
        return 0.0;
    }

    let tau = std::f64::consts::PI * 2.0;
    let mut best = -1.0;
    for step in 0..phase_steps {
        let phase = tau * step as f64 / phase_steps as f64;
        let wave: Vec<f64> = (0..signal.len())
            .map(|idx| {
                let angle = tau * frequency as f64 * idx as f64 / signal.len() as f64 + phase;
                angle.sin().max(0.0)
            })
            .collect();
        let corr = pearson_correlation(signal, &wave);
        if corr > best {
            best = corr;
        }
    }

    best
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;

    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }

    if denom_a == 0.0 || denom_b == 0.0 {
        return 0.0;
    }

    num / (denom_a.sqrt() * denom_b.sqrt())
}

/// Verify that the model actually learned modular addition
/// by testing it on ALL possible examples systematically
pub fn verify_full_accuracy<B: Backend>(
    model: &Transformer<B>,
    device: &B::Device,
) -> (f32, f32) {
    println!("🔍 Verifying model on COMPLETE dataset (no sampling)...");

    let train_dataset = ModularAdditionDataset::new(true, 42);
    let val_dataset = ModularAdditionDataset::new(false, 42);

    let train_acc = test_all_examples(model, &train_dataset, device);
    let val_acc = test_all_examples(model, &val_dataset, device);

    println!("📊 FULL Dataset Results:");
    println!("   Training: {}/{} = {:.2}%",
        (train_acc * train_dataset.len() as f32) as usize,
        train_dataset.len(),
        train_acc * 100.0);
    println!("   Validation: {}/{} = {:.2}%",
        (val_acc * val_dataset.len() as f32) as usize,
        val_dataset.len(),
        val_acc * 100.0);

    (train_acc, val_acc)
}

/// Test model on ALL examples in dataset (no random sampling)
fn test_all_examples<B: Backend>(
    model: &Transformer<B>,
    dataset: &ModularAdditionDataset,
    device: &B::Device,
) -> f32 {
    let mut total_correct = 0;
    let total = dataset.len();

    // Test in batches to avoid memory issues
    let batch_size = 100;
    for batch_start in (0..total).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(total);
        let batch_len = batch_end - batch_start;

        let mut inputs_vec = Vec::new();
        let mut targets_vec = Vec::new();

        for idx in batch_start..batch_end {
            let (input, target) = dataset.get(idx).unwrap();
            inputs_vec.extend(input);
            targets_vec.push(target as i32);
        }

        let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
            .reshape([batch_len, 3]);

        let logits = model.forward(inputs);
        let predictions = logits.argmax(1).squeeze::<1>();
        let predictions_vec: Vec<i32> = predictions.into_data().to_vec().unwrap();

        let correct = predictions_vec
            .iter()
            .zip(targets_vec.iter())
            .filter(|(p, t)| p == t)
            .count();

        total_correct += correct;
    }

    total_correct as f32 / total as f32
}

/// Test specific examples to see if model actually computes mod p
pub fn test_specific_examples<B: Backend>(
    model: &Transformer<B>,
    device: &B::Device,
) {
    println!();
    println!("🧪 Testing specific examples:");

    let modulus = ModularAdditionDataset::modulus();
    let equals_token = ModularAdditionDataset::equals_token();
    let test_cases = vec![
        (0, 0, 0),
        (1, 1, 2),
        (50, 50, (50 + 50) % modulus),
        (112, 1, (112 + 1) % modulus),
        (56, 57, (56 + 57) % modulus),
        (112, 112, (112 + 112) % modulus),
        (10, 20, 30),
        (60, 60, (60 + 60) % modulus),
    ];

    for (a, b, expected) in test_cases {
        let input_vec = vec![a as i32, b as i32, equals_token as i32];
        let input = Tensor::<B, 1, Int>::from_ints(input_vec.as_slice(), device)
            .reshape([1, 3]);

        let logits = model.forward(input);
        let prediction = logits.argmax(1).squeeze::<1>();
        let pred_value: i32 = prediction.into_data().to_vec().unwrap()[0];

        let correct = if pred_value == expected as i32 { "✓" } else { "✗" };
        println!(
            "   {} + {} mod {} = {} | Model: {} {}",
            a, b, modulus, expected, pred_value, correct
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_histories(
        train_acc: &[(usize, f32)],
        val_acc: &[(usize, f32)],
        val_loss: &[(usize, f64)],
    ) -> (LossHistory, AccuracyHistory) {
        let mut loss_history = LossHistory::new();
        loss_history.val_snapshots = val_loss.to_vec();
        let mut accuracy_history = AccuracyHistory::new();
        accuracy_history.train_snapshots = train_acc.to_vec();
        accuracy_history.val_snapshots = val_acc.to_vec();
        (loss_history, accuracy_history)
    }

    #[test]
    fn grokking_phase_verification_passes_on_synthetic_signal() {
        let train_acc = vec![(0, 0.2), (500, 0.995), (1000, 0.999)];
        let val_acc = vec![
            (0, 0.01),
            (500, 0.01),
            (1500, 0.01),
            (2200, 0.995),
            (2500, 0.999),
        ];
        let val_loss = vec![
            (0, 2.0),
            (500, 2.0),
            (1500, 2.0),
            (2200, 0.2),
            (2500, 0.15),
        ];

        let (loss_history, accuracy_history) = make_histories(&train_acc, &val_acc, &val_loss);
        let config = GrokkingVerificationConfig {
            early_step_max: 1000,
            plateau_min_step: 1500,
            generalization_window: 1500,
            loss_window: 2,
            ..GrokkingVerificationConfig::default_for_modulus(ModularAdditionDataset::modulus())
        };

        let report = verify_grokking_phase_transition(&loss_history, &accuracy_history, &config)
            .expect("expected grokking verification to pass");
        assert!(report.transition_step >= config.plateau_min_step);
        assert!(report.early_train_step <= config.early_step_max);
    }

    #[test]
    fn grokking_phase_verification_fails_without_transition() {
        let train_acc = vec![(0, 0.2), (500, 0.995)];
        let val_acc = vec![(0, 0.01), (500, 0.01), (1500, 0.02)];
        let val_loss = vec![(0, 2.0), (500, 2.0), (1500, 1.9)];

        let (loss_history, accuracy_history) = make_histories(&train_acc, &val_acc, &val_loss);
        let config = GrokkingVerificationConfig {
            early_step_max: 1000,
            plateau_min_step: 1000,
            generalization_window: 1000,
            loss_window: 2,
            ..GrokkingVerificationConfig::default_for_modulus(ModularAdditionDataset::modulus())
        };

        let err = verify_grokking_phase_transition(&loss_history, &accuracy_history, &config)
            .expect_err("expected grokking verification to fail");
        assert!(err.contains("validation accuracy never reaches target"));
    }

    #[test]
    fn restricted_loss_detection_passes_with_early_drop() {
        let full_loss = vec![
            (0, 2.0),
            (500, 2.0),
            (1500, 2.0),
            (2500, 2.0),
            (3500, 0.3),
        ];
        let restricted_loss = vec![
            (0, 2.0),
            (500, 2.0),
            (1500, 0.8),
            (2500, 0.4),
            (3500, 0.3),
        ];

        let config = RestrictedLossVerificationConfig {
            plateau_min_step: 1000,
            drop_window: 2,
            drop_fraction: 0.5,
            min_step_lead: 500,
        };

        let report = verify_restricted_loss_early_drop(&full_loss, &restricted_loss, &config)
            .expect("expected restricted loss verification to pass");
        assert!(report.restricted_drop_step < report.full_drop_step);
    }

    #[test]
    fn restricted_loss_detection_fails_when_full_drops_first() {
        let full_loss = vec![(0, 2.0), (1000, 1.0), (1500, 0.8)];
        let restricted_loss = vec![(0, 2.0), (1000, 2.0), (1500, 0.8)];

        let config = RestrictedLossVerificationConfig {
            plateau_min_step: 500,
            drop_window: 1,
            drop_fraction: 0.3,
            min_step_lead: 200,
        };

        let err = verify_restricted_loss_early_drop(&full_loss, &restricted_loss, &config)
            .expect_err("expected restricted loss verification to fail");
        assert!(err.contains("restricted loss drops"));
    }

    #[test]
    fn excluded_loss_spike_verification_passes() {
        let excluded_loss = vec![(0, 1.0), (1500, 2.2), (3000, 1.1)];
        let config = ExcludedLossSpikeConfig {
            min_relative_increase: 0.5,
        };

        let report = verify_excluded_loss_spike(&excluded_loss, &config)
            .expect("expected excluded loss spike verification to pass");
        assert_eq!(report.spike_step, 1500);
    }

    #[test]
    fn excluded_loss_spike_verification_fails_without_spike() {
        let excluded_loss = vec![(0, 1.0), (1500, 1.1), (3000, 1.0)];
        let config = ExcludedLossSpikeConfig {
            min_relative_increase: 0.5,
        };

        let err = verify_excluded_loss_spike(&excluded_loss, &config)
            .expect_err("expected excluded loss spike verification to fail");
        assert!(err.contains("excluded loss spike too small"));
    }

    #[test]
    fn mlp_activation_wave_verification_passes_on_rectified_sine() {
        let n = ModularAdditionDataset::modulus();
        let freq = 5.0;
        let mut signal = Vec::with_capacity(n);
        for idx in 0..n {
            let angle = 2.0 * std::f64::consts::PI * freq * idx as f64 / n as f64;
            let rectified = angle.sin().max(0.0);
            let wobble = (2.0 * std::f64::consts::PI * 2.0 * idx as f64 / n as f64)
                .sin()
                .abs()
                * 0.02;
            signal.push(rectified + wobble);
        }

        let config = MlpWaveVerificationConfig {
            max_negative_fraction: 0.0,
            negative_tolerance: 1e-6,
            min_dominant_ratio: 3.0,
            min_correlation: 0.7,
            phase_steps: 32,
        };

        let report =
            verify_mlp_activation_waves_from_signals(&[(0, signal)], &config)
                .expect("expected rectified sine to pass");
        assert_eq!(report.neuron_reports.len(), 1);
    }

    #[test]
    fn mlp_activation_wave_verification_fails_on_noise() {
        let n = ModularAdditionDataset::modulus();
        let signal: Vec<f64> = (0..n).map(|idx| idx as f64 / n as f64).collect();

        let config = MlpWaveVerificationConfig {
            max_negative_fraction: 0.0,
            negative_tolerance: 1e-6,
            min_dominant_ratio: 4.0,
            min_correlation: 0.8,
            phase_steps: 16,
        };

        let err =
            verify_mlp_activation_waves_from_signals(&[(0, signal)], &config)
                .expect_err("expected linear signal to fail");
        assert!(err.contains("dominant frequency ratio") || err.contains("correlation"));
    }
}
