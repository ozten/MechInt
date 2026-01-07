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
    pub accuracy_window: usize,
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
            accuracy_window: 5,
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

#[derive(Debug, Clone)]
pub struct PairwiseManifoldConfig {
    pub fixed_y: usize,
    pub neuron_indices: Vec<usize>,
    pub min_post_score: f64,
    pub max_pre_score: f64,
    pub max_axis_ratio: f64,
    pub min_pairs_passing: usize,
}

impl PairwiseManifoldConfig {
    pub fn default_for_modulus(_modulus: usize) -> Self {
        Self {
            fixed_y: 0,
            neuron_indices: (0..7).collect(),
            min_post_score: 0.75,
            max_pre_score: 0.55,
            max_axis_ratio: 8.0,
            min_pairs_passing: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PairwiseManifoldPairReport {
    pub neuron_a: usize,
    pub neuron_b: usize,
    pub circularity_score: f64,
    pub radius_cv: f64,
    pub axis_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct PairwiseManifoldReport {
    pub pair_reports: Vec<PairwiseManifoldPairReport>,
    pub average_score: f64,
}

#[derive(Debug, Clone)]
pub struct PairwiseManifoldTransitionReport {
    pub pre_average_score: f64,
    pub post_average_score: f64,
    pub passing_pairs: usize,
    pub total_pairs: usize,
}

#[derive(Debug, Clone)]
pub struct DiagonalRidgeConfig {
    pub neuron_index: usize,
    pub max_lag: usize,
    pub min_diagonal_ratio: f64,
    pub min_diagonal_score: f64,
}

impl DiagonalRidgeConfig {
    pub fn default_for_modulus(modulus: usize) -> Self {
        let max_lag = (modulus / 10).max(4).min(20);
        Self {
            neuron_index: 10,
            max_lag,
            min_diagonal_ratio: 1.3,
            min_diagonal_score: 0.2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagonalRidgeReport {
    pub neuron_index: usize,
    pub diagonal_score: f64,
    pub axis_score: f64,
    pub ratio: f64,
}

#[derive(Debug, Clone)]
pub struct SnakeCurveConfig {
    pub early_step_max: usize,
    pub train_convergence_threshold: f32,
    pub val_plateau_min_step: usize,
    pub val_plateau_max_acc: f32,
    pub val_final_threshold: f32,
}

impl SnakeCurveConfig {
    pub fn default_for_modulus(modulus: usize) -> Self {
        Self {
            early_step_max: 1000,
            train_convergence_threshold: 0.95,
            val_plateau_min_step: 1000,
            val_plateau_max_acc: 1.0 / modulus as f32 + 0.05,
            val_final_threshold: 0.90,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WeightNormDecayConfig {
    pub min_relative_decrease: f64,
    pub plateau_min_step: usize,
    pub grok_step_window: usize,
}

impl WeightNormDecayConfig {
    pub fn default_for_grokking() -> Self {
        Self {
            min_relative_decrease: 0.1,
            plateau_min_step: 1000,
            grok_step_window: 500,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WeightNormDecayReport {
    pub initial_norm: f64,
    pub final_norm: f64,
    pub relative_decrease: f64,
    pub grok_step_norm: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SnakeCurveReport {
    pub train_converged_by_step: usize,
    pub val_plateau_max_acc: f32,
    pub val_grok_step: usize,
    pub curve_shape_valid: bool,
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

    verify_accuracy_jump(
        &accuracy_history.val_snapshots,
        *transition_step,
        config.accuracy_window,
        config.chance_accuracy,
        config.chance_tolerance,
        config.target_val_acc_threshold,
    )?;

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

fn verify_accuracy_jump(
    val_accuracy: &[(usize, f32)],
    transition_step: usize,
    window: usize,
    chance_accuracy: f32,
    chance_tolerance: f32,
    target_val_acc_threshold: f32,
) -> Result<(), String> {
    let before_values: Vec<f32> = val_accuracy
        .iter()
        .filter(|(step, _)| *step < transition_step)
        .map(|(_, acc)| *acc)
        .collect();

    let after_values: Vec<f32> = val_accuracy
        .iter()
        .filter(|(step, _)| *step >= transition_step)
        .map(|(_, acc)| *acc)
        .collect();

    if before_values.len() < window || after_values.len() < window {
        return Err("insufficient accuracy samples around transition".to_string());
    }

    let before_avg: f32 =
        before_values[before_values.len() - window..].iter().sum::<f32>() / window as f32;
    let after_avg: f32 = after_values[..window].iter().sum::<f32>() / window as f32;

    if before_avg > chance_accuracy + chance_tolerance {
        return Err(format!(
            "validation accuracy plateau too high before transition (avg {:.4})",
            before_avg
        ));
    }

    if after_avg < target_val_acc_threshold {
        return Err(format!(
            "validation accuracy after transition too low (avg {:.4})",
            after_avg
        ));
    }

    Ok(())
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

pub fn verify_mlp_pairwise_manifold_transition<B: Backend>(
    pre_model: &Transformer<B>,
    post_model: &Transformer<B>,
    device: &B::Device,
    config: &PairwiseManifoldConfig,
) -> Result<PairwiseManifoldTransitionReport, String> {
    let modulus = ModularAdditionDataset::modulus();
    if config.fixed_y >= modulus {
        return Err(format!(
            "fixed_y {} out of range (modulus {})",
            config.fixed_y, modulus
        ));
    }

    let pre_activations = collect_mlp_post_relu_activations(pre_model, device, config.fixed_y)?;
    let post_activations = collect_mlp_post_relu_activations(post_model, device, config.fixed_y)?;

    let d_ff = pre_activations.len();
    if post_activations.len() != d_ff {
        return Err("post activations shape mismatch".to_string());
    }

    if config.neuron_indices.len() < 2 {
        return Err("need at least two neuron indices".to_string());
    }

    for &idx in &config.neuron_indices {
        if idx >= d_ff {
            return Err(format!(
                "neuron index {} out of range (d_ff={})",
                idx, d_ff
            ));
        }
    }

    let pre_signals: Vec<(usize, Vec<f64>)> = config
        .neuron_indices
        .iter()
        .map(|&idx| (idx, pre_activations[idx].clone()))
        .collect();
    let post_signals: Vec<(usize, Vec<f64>)> = config
        .neuron_indices
        .iter()
        .map(|&idx| (idx, post_activations[idx].clone()))
        .collect();

    verify_pairwise_manifold_transition_from_signals(&pre_signals, &post_signals, config)
}

pub fn collect_mlp_activation_surface<B: Backend>(
    model: &Transformer<B>,
    device: &B::Device,
    neuron_index: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let modulus = ModularAdditionDataset::modulus();
    let equals_token = ModularAdditionDataset::equals_token();

    let mut inputs_vec = Vec::with_capacity(modulus * modulus * 3);
    for x in 0..modulus {
        for y in 0..modulus {
            inputs_vec.push(x as i32);
            inputs_vec.push(y as i32);
            inputs_vec.push(equals_token as i32);
        }
    }

    let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
        .reshape([modulus * modulus, 3]);
    let (_, mlp_acts) = model.forward_with_mlp_activations(inputs);
    let [batch_size, d_ff] = mlp_acts.dims();
    if batch_size != modulus * modulus {
        return Err(format!(
            "unexpected activation batch size {} (expected {})",
            batch_size,
            modulus * modulus
        ));
    }
    if neuron_index >= d_ff {
        return Err(format!(
            "neuron index {} out of range (d_ff={})",
            neuron_index, d_ff
        ));
    }

    let data: Vec<f32> = mlp_acts.into_data().to_vec().unwrap();
    let mut surface = vec![vec![0.0f64; modulus]; modulus];
    for x in 0..modulus {
        for y in 0..modulus {
            let batch = x * modulus + y;
            let idx = batch * d_ff + neuron_index;
            surface[x][y] = data[idx] as f64;
        }
    }

    Ok(surface)
}

pub fn verify_constructive_interference_surface_from_grid(
    surface: &[Vec<f64>],
    config: &DiagonalRidgeConfig,
) -> Result<DiagonalRidgeReport, String> {
    let (diag_main, diag_anti, axis_x, axis_y) =
        surface_autocorrelation_scores(surface, config.max_lag)?;

    let diagonal_score = diag_main.max(diag_anti);
    let axis_score = axis_x.max(axis_y);
    let ratio = if axis_score > 0.0 {
        diagonal_score / axis_score
    } else {
        f64::INFINITY
    };

    if diagonal_score < config.min_diagonal_score {
        return Err(format!(
            "diagonal autocorrelation {:.3} below threshold {:.3}",
            diagonal_score, config.min_diagonal_score
        ));
    }

    if ratio < config.min_diagonal_ratio {
        return Err(format!(
            "diagonal/axis ratio {:.3} below threshold {:.3}",
            ratio, config.min_diagonal_ratio
        ));
    }

    Ok(DiagonalRidgeReport {
        neuron_index: config.neuron_index,
        diagonal_score,
        axis_score,
        ratio,
    })
}

pub fn verify_pairwise_manifold_transition_from_signals(
    pre_signals: &[(usize, Vec<f64>)],
    post_signals: &[(usize, Vec<f64>)],
    config: &PairwiseManifoldConfig,
) -> Result<PairwiseManifoldTransitionReport, String> {
    let pre_report = evaluate_pairwise_manifolds_from_signals(pre_signals)?;
    let post_report = evaluate_pairwise_manifolds_from_signals(post_signals)?;

    if pre_report.average_score > config.max_pre_score {
        return Err(format!(
            "pre-grok average circularity {:.3} exceeds threshold {:.3}",
            pre_report.average_score, config.max_pre_score
        ));
    }

    let passing_pairs = post_report
        .pair_reports
        .iter()
        .filter(|pair| {
            pair.circularity_score >= config.min_post_score
                && pair.axis_ratio <= config.max_axis_ratio
        })
        .count();
    if passing_pairs < config.min_pairs_passing {
        return Err(format!(
            "post-grok only {} pairs meet circularity threshold (need {})",
            passing_pairs, config.min_pairs_passing
        ));
    }

    Ok(PairwiseManifoldTransitionReport {
        pre_average_score: pre_report.average_score,
        post_average_score: post_report.average_score,
        passing_pairs,
        total_pairs: post_report.pair_reports.len(),
    })
}

fn evaluate_pairwise_manifolds_from_signals(
    signals: &[(usize, Vec<f64>)],
) -> Result<PairwiseManifoldReport, String> {
    if signals.len() < 2 {
        return Err("need at least two activation signals".to_string());
    }

    let signal_len = signals[0].1.len();
    if signal_len < 3 {
        return Err("activation signal too short".to_string());
    }

    for (idx, signal) in signals {
        if signal.len() != signal_len {
            return Err(format!(
                "signal length mismatch for neuron {}",
                idx
            ));
        }
    }

    let mut reports = Vec::new();
    for i in 0..signals.len() {
        for j in (i + 1)..signals.len() {
            let (neuron_a, signal_a) = &signals[i];
            let (neuron_b, signal_b) = &signals[j];
            let points: Vec<(f64, f64)> = signal_a
                .iter()
                .zip(signal_b.iter())
                .map(|(x, y)| (*x, *y))
                .collect();
            let (score, radius_cv, axis_ratio) =
                match pairwise_circularity_metrics(&points) {
                    Ok(metrics) => metrics,
                    Err(_) => (0.0, 1.0, f64::INFINITY),
                };
            reports.push(PairwiseManifoldPairReport {
                neuron_a: *neuron_a,
                neuron_b: *neuron_b,
                circularity_score: score,
                radius_cv,
                axis_ratio,
            });
        }
    }

    let average_score = reports
        .iter()
        .map(|pair| pair.circularity_score)
        .sum::<f64>()
        / reports.len() as f64;

    Ok(PairwiseManifoldReport {
        pair_reports: reports,
        average_score,
    })
}

fn pairwise_circularity_metrics(points: &[(f64, f64)]) -> Result<(f64, f64, f64), String> {
    if points.len() < 3 {
        return Err("not enough points for circularity".to_string());
    }

    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n;

    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov_xy = 0.0;
    for (x, y) in points {
        let dx = x - mean_x;
        let dy = y - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }
    var_x /= n;
    var_y /= n;
    cov_xy /= n;

    let trace = var_x + var_y;
    let det = var_x * var_y - cov_xy * cov_xy;
    if det <= 1.0e-12 || trace <= 0.0 {
        return Err("degenerate covariance".to_string());
    }

    let half_trace = trace * 0.5;
    let disc = (half_trace * half_trace - det).max(0.0).sqrt();
    let lambda1 = half_trace + disc;
    let lambda2 = half_trace - disc;
    if lambda2 <= 0.0 {
        return Err("degenerate eigenvalues".to_string());
    }

    let axis_ratio = (lambda1 / lambda2).sqrt();

    let mut v1_x = cov_xy;
    let mut v1_y = lambda1 - var_x;
    if v1_x.abs() < 1.0e-12 && v1_y.abs() < 1.0e-12 {
        v1_x = 1.0;
        v1_y = 0.0;
    }
    let norm = (v1_x * v1_x + v1_y * v1_y).sqrt();
    v1_x /= norm;
    v1_y /= norm;
    let v2_x = -v1_y;
    let v2_y = v1_x;

    let mut radii = Vec::with_capacity(points.len());
    for (x, y) in points {
        let dx = x - mean_x;
        let dy = y - mean_y;
        let p1 = dx * v1_x + dy * v1_y;
        let p2 = dx * v2_x + dy * v2_y;
        let w1 = p1 / lambda1.sqrt();
        let w2 = p2 / lambda2.sqrt();
        radii.push((w1 * w1 + w2 * w2).sqrt());
    }

    let mean_radius = radii.iter().sum::<f64>() / radii.len() as f64;
    if mean_radius == 0.0 {
        return Err("zero mean radius".to_string());
    }

    let mut variance = 0.0;
    for r in &radii {
        let diff = r - mean_radius;
        variance += diff * diff;
    }
    let std = (variance / radii.len() as f64).sqrt();
    let radius_cv = std / mean_radius;
    let score = 1.0 / (1.0 + radius_cv);

    Ok((score, radius_cv, axis_ratio))
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

fn surface_autocorrelation_scores(
    surface: &[Vec<f64>],
    max_lag: usize,
) -> Result<(f64, f64, f64, f64), String> {
    let (height, width) = surface_dimensions(surface)?;
    if height < 2 || width < 2 {
        return Err("surface grid too small".to_string());
    }

    let (mean, var) = surface_mean_variance(surface)?;
    if var == 0.0 {
        return Err("surface variance is zero".to_string());
    }

    let max_lag = max_lag.min(height.saturating_sub(1)).min(width.saturating_sub(1));
    if max_lag == 0 {
        return Err("max_lag too small for surface".to_string());
    }

    let diag_main = average_autocorrelation(surface, mean, var, max_lag, 1, 1)?;
    let diag_anti = average_autocorrelation(surface, mean, var, max_lag, 1, -1)?;
    let axis_x = average_autocorrelation(surface, mean, var, max_lag, 1, 0)?;
    let axis_y = average_autocorrelation(surface, mean, var, max_lag, 0, 1)?;

    Ok((diag_main, diag_anti, axis_x, axis_y))
}

fn surface_dimensions(surface: &[Vec<f64>]) -> Result<(usize, usize), String> {
    let height = surface.len();
    if height == 0 {
        return Err("surface is empty".to_string());
    }
    let width = surface[0].len();
    if width == 0 {
        return Err("surface has empty rows".to_string());
    }
    for (idx, row) in surface.iter().enumerate() {
        if row.len() != width {
            return Err(format!(
                "surface row {} has length {}, expected {}",
                idx,
                row.len(),
                width
            ));
        }
    }
    Ok((height, width))
}

fn surface_mean_variance(surface: &[Vec<f64>]) -> Result<(f64, f64), String> {
    let (height, width) = surface_dimensions(surface)?;
    let mut sum = 0.0;
    for row in surface {
        for value in row {
            sum += *value;
        }
    }
    let count = (height * width) as f64;
    let mean = sum / count;
    let mut variance = 0.0;
    for row in surface {
        for value in row {
            let diff = *value - mean;
            variance += diff * diff;
        }
    }
    variance /= count;
    Ok((mean, variance))
}

fn average_autocorrelation(
    surface: &[Vec<f64>],
    mean: f64,
    variance: f64,
    max_lag: usize,
    dx: i32,
    dy: i32,
) -> Result<f64, String> {
    let (height, width) = surface_dimensions(surface)?;
    let mut total = 0.0;
    let mut lag_count = 0usize;

    for lag in 1..=max_lag {
        let mut sum = 0.0;
        let mut count = 0usize;
        let lag = lag as i32;
        for x in 0..height {
            for y in 0..width {
                let x2 = x as i32 + dx * lag;
                let y2 = y as i32 + dy * lag;
                if x2 < 0 || y2 < 0 {
                    continue;
                }
                let x2 = x2 as usize;
                let y2 = y2 as usize;
                if x2 >= height || y2 >= width {
                    continue;
                }
                let a = surface[x][y] - mean;
                let b = surface[x2][y2] - mean;
                sum += a * b;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let corr = sum / (count as f64 * variance);
        total += corr.abs();
        lag_count += 1;
    }

    if lag_count == 0 {
        return Err("no valid lags for autocorrelation".to_string());
    }

    Ok(total / lag_count as f64)
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

/// Verify that weight norms decrease over time with weight decay
/// and show correlation with the grokking transition
pub fn verify_weight_norm_decay(
    weight_norm_history: &[(usize, f64)],
    grok_step: Option<usize>,
    config: &WeightNormDecayConfig,
) -> Result<WeightNormDecayReport, String> {
    if weight_norm_history.len() < 2 {
        return Err("weight norm history requires at least 2 snapshots".to_string());
    }

    let initial_norm = weight_norm_history
        .first()
        .ok_or_else(|| "weight norm history is empty".to_string())?
        .1;
    let final_norm = weight_norm_history
        .last()
        .ok_or_else(|| "weight norm history is empty".to_string())?
        .1;

    if initial_norm <= 0.0 {
        return Err("initial weight norm must be positive".to_string());
    }

    // Check that norms are generally decreasing (with weight decay)
    let relative_decrease = (initial_norm - final_norm) / initial_norm;
    if relative_decrease < config.min_relative_decrease {
        return Err(format!(
            "weight norm decrease {:.4} below threshold {:.4} (initial {:.4}, final {:.4})",
            relative_decrease, config.min_relative_decrease, initial_norm, final_norm
        ));
    }

    // Check that norms are not increasing (weight decay should monotonically decrease norms)
    for window in weight_norm_history.windows(2) {
        let prev_norm = window[0].1;
        let next_norm = window[1].1;
        if next_norm > prev_norm * 1.01 {
            // Allow 1% tolerance for numerical noise
            return Err(format!(
                "weight norm increased from {:.4} to {:.4} at step {}",
                prev_norm, next_norm, window[1].0
            ));
        }
    }

    // If grok step is provided, check that norms show a shift around grokking
    let grok_step_norm = if let Some(step) = grok_step {
        weight_norm_history
            .iter()
            .filter(|(s, _)| {
                *s >= step.saturating_sub(config.grok_step_window)
                    && *s <= step + config.grok_step_window
            })
            .map(|(_, norm)| *norm)
            .next()
    } else {
        None
    };

    Ok(WeightNormDecayReport {
        initial_norm,
        final_norm,
        relative_decrease,
        grok_step_norm,
    })
}

/// Verify the characteristic "snake curve" shape of grokking phenomenon
/// on log-scale plots: train accuracy rises early, validation stays flat then jumps
pub fn verify_snake_curve_shape(
    train_acc: &[(usize, f32)],
    val_acc: &[(usize, f32)],
    config: &SnakeCurveConfig,
) -> Result<SnakeCurveReport, String> {
    if train_acc.is_empty() || val_acc.is_empty() {
        return Err("empty accuracy history".to_string());
    }

    // Phase 1: Verify early train convergence (immediate drop on log scale)
    let train_converged_step = train_acc
        .iter()
        .find(|(_, acc)| *acc >= config.train_convergence_threshold)
        .map(|(step, _)| *step)
        .ok_or_else(|| {
            format!(
                "train accuracy never reaches {} threshold",
                config.train_convergence_threshold
            )
        })?;

    if train_converged_step > config.early_step_max {
        return Err(format!(
            "train accuracy converges at step {}, expected before {}",
            train_converged_step, config.early_step_max
        ));
    }

    // Phase 2: Verify validation plateau (flat on log scale)
    let plateau_samples: Vec<f32> = val_acc
        .iter()
        .filter(|(step, _)| *step >= 1 && *step <= config.val_plateau_min_step)
        .map(|(_, acc)| *acc)
        .collect();

    if plateau_samples.is_empty() {
        return Err("no validation samples in plateau region".to_string());
    }

    let val_plateau_max = plateau_samples
        .iter()
        .fold(0.0f32, |max_val, &acc| max_val.max(acc));

    if val_plateau_max > config.val_plateau_max_acc {
        return Err(format!(
            "validation accuracy in plateau region too high: max {:.4} exceeds {:.4}",
            val_plateau_max, config.val_plateau_max_acc
        ));
    }

    // Phase 3: Verify delayed validation jump (crash up on log scale)
    let val_grok_step = val_acc
        .iter()
        .filter(|(step, _)| *step > config.val_plateau_min_step)
        .find(|(_, acc)| *acc >= config.val_final_threshold)
        .map(|(step, _)| *step)
        .ok_or_else(|| {
            format!(
                "validation accuracy never reaches {} after plateau",
                config.val_final_threshold
            )
        })?;

    // Phase 4: Verify phase ordering (snake shape: early train, late val)
    let curve_shape_valid = train_converged_step < config.val_plateau_min_step
        && val_grok_step > config.val_plateau_min_step
        && train_converged_step < val_grok_step;

    if !curve_shape_valid {
        return Err(format!(
            "invalid curve shape: train converged at {}, val plateau until {}, val grokked at {}",
            train_converged_step, config.val_plateau_min_step, val_grok_step
        ));
    }

    Ok(SnakeCurveReport {
        train_converged_by_step: train_converged_step,
        val_plateau_max_acc: val_plateau_max,
        val_grok_step,
        curve_shape_valid,
    })
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
            accuracy_window: 2,
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
            accuracy_window: 2,
            loss_window: 2,
            ..GrokkingVerificationConfig::default_for_modulus(ModularAdditionDataset::modulus())
        };

        let err = verify_grokking_phase_transition(&loss_history, &accuracy_history, &config)
            .expect_err("expected grokking verification to fail");
        assert!(err.contains("validation accuracy never reaches target"));
    }

    #[test]
    fn grokking_phase_verification_fails_with_slow_accuracy_rise() {
        let train_acc = vec![(0, 0.2), (500, 0.995), (1000, 0.999)];
        let val_acc = vec![
            (0, 0.01),
            (500, 0.01),
            (1500, 0.5),
            (2000, 0.6),
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
            plateau_min_step: 1000,
            generalization_window: 2000,
            accuracy_window: 2,
            loss_window: 2,
            ..GrokkingVerificationConfig::default_for_modulus(ModularAdditionDataset::modulus())
        };

        let err = verify_grokking_phase_transition(&loss_history, &accuracy_history, &config)
            .expect_err("expected grokking verification to fail");
        assert!(err.contains("plateau too high"));
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

    #[test]
    fn pairwise_manifold_transition_passes_on_elliptic_signals() {
        let n = ModularAdditionDataset::modulus();
        let freq = 5.0;
        let mut post_signals = Vec::new();
        for neuron_idx in 0..7usize {
            let phase = 2.0 * std::f64::consts::PI * neuron_idx as f64 / 7.0;
            let signal: Vec<f64> = (0..n)
                .map(|idx| {
                    let angle = 2.0 * std::f64::consts::PI * freq * idx as f64 / n as f64 + phase;
                    angle.sin()
                })
                .collect();
            post_signals.push((neuron_idx, signal));
        }

        let mut pre_signals = Vec::new();
        for neuron_idx in 0..7usize {
            let signal: Vec<f64> = (0..n)
                .map(|idx| idx as f64 / n as f64 + neuron_idx as f64 * 0.01)
                .collect();
            pre_signals.push((neuron_idx, signal));
        }

        let config = PairwiseManifoldConfig {
            fixed_y: 0,
            neuron_indices: (0..7).collect(),
            min_post_score: 0.75,
            max_pre_score: 0.55,
            max_axis_ratio: 8.0,
            min_pairs_passing: 8,
        };

        let report = verify_pairwise_manifold_transition_from_signals(
            &pre_signals,
            &post_signals,
            &config,
        )
        .expect("expected pairwise manifold transition to pass");
        assert!(report.passing_pairs >= config.min_pairs_passing);
    }

    #[test]
    fn pairwise_manifold_transition_fails_without_rings() {
        let n = ModularAdditionDataset::modulus();
        let mut signals = Vec::new();
        for neuron_idx in 0..7usize {
            let signal: Vec<f64> = (0..n)
                .map(|idx| idx as f64 / n as f64 + neuron_idx as f64 * 0.02)
                .collect();
            signals.push((neuron_idx, signal));
        }

        let config = PairwiseManifoldConfig {
            fixed_y: 0,
            neuron_indices: (0..7).collect(),
            min_post_score: 0.8,
            max_pre_score: 0.55,
            max_axis_ratio: 4.0,
            min_pairs_passing: 5,
        };

        let err = verify_pairwise_manifold_transition_from_signals(
            &signals,
            &signals,
            &config,
        )
        .expect_err("expected pairwise manifold transition to fail");
        assert!(err.contains("post-grok") || err.contains("pre-grok"));
    }

    #[test]
    fn diagonal_ridge_verification_passes_on_diagonal_corrugation() {
        let n = 50;
        let mut surface = vec![vec![0.0; n]; n];

        // Create diagonal ridges: f(x,y) = sin(2π * freq * (x+y) / n)
        let freq = 3.0;
        for x in 0..n {
            for y in 0..n {
                let angle = 2.0 * std::f64::consts::PI * freq * (x + y) as f64 / n as f64;
                surface[x][y] = angle.sin();
            }
        }

        let config = DiagonalRidgeConfig {
            neuron_index: 10,
            max_lag: 8,
            min_diagonal_ratio: 1.3,
            min_diagonal_score: 0.2,
        };

        let report = verify_constructive_interference_surface_from_grid(&surface, &config)
            .expect("expected diagonal ridge verification to pass");
        assert!(report.diagonal_score > report.axis_score);
        assert!(report.ratio >= config.min_diagonal_ratio);
        assert!(report.diagonal_score >= config.min_diagonal_score);
    }

    #[test]
    fn diagonal_ridge_verification_passes_on_anti_diagonal_pattern() {
        let n = 50;
        let mut surface = vec![vec![0.0; n]; n];

        // Create anti-diagonal ridges: f(x,y) = sin(2π * freq * (x-y) / n)
        let freq = 3.0;
        for x in 0..n {
            for y in 0..n {
                let angle = 2.0 * std::f64::consts::PI * freq * (x as i32 - y as i32) as f64 / n as f64;
                surface[x][y] = angle.sin();
            }
        }

        let config = DiagonalRidgeConfig {
            neuron_index: 10,
            max_lag: 8,
            min_diagonal_ratio: 1.3,
            min_diagonal_score: 0.2,
        };

        let report = verify_constructive_interference_surface_from_grid(&surface, &config)
            .expect("expected anti-diagonal ridge verification to pass");
        assert!(report.diagonal_score > report.axis_score);
        assert!(report.ratio >= config.min_diagonal_ratio);
    }

    #[test]
    fn diagonal_ridge_verification_fails_on_axis_aligned_pattern() {
        let n = 50;
        let mut surface = vec![vec![0.0; n]; n];

        // Create axis-aligned ridges: f(x,y) = sin(2π * freq * x / n)
        let freq = 3.0;
        for x in 0..n {
            for y in 0..n {
                let angle = 2.0 * std::f64::consts::PI * freq * x as f64 / n as f64;
                surface[x][y] = angle.sin();
            }
        }

        let config = DiagonalRidgeConfig {
            neuron_index: 10,
            max_lag: 8,
            min_diagonal_ratio: 1.3,
            min_diagonal_score: 0.2,
        };

        let err = verify_constructive_interference_surface_from_grid(&surface, &config)
            .expect_err("expected axis-aligned pattern to fail diagonal verification");
        assert!(err.contains("diagonal/axis ratio") || err.contains("diagonal autocorrelation"));
    }

    #[test]
    fn diagonal_ridge_verification_fails_on_flat_surface() {
        let n = 50;
        let surface = vec![vec![1.5; n]; n];

        let config = DiagonalRidgeConfig {
            neuron_index: 10,
            max_lag: 8,
            min_diagonal_ratio: 1.3,
            min_diagonal_score: 0.2,
        };

        let err = verify_constructive_interference_surface_from_grid(&surface, &config)
            .expect_err("expected flat surface to fail");
        assert!(err.contains("surface variance is zero"));
    }

    #[test]
    fn diagonal_ridge_verification_fails_on_random_noise() {
        let n = 50;
        let mut surface = vec![vec![0.0; n]; n];

        // Create random noise (low autocorrelation everywhere)
        for x in 0..n {
            for y in 0..n {
                surface[x][y] = ((x * 17 + y * 31) % 100) as f64 / 100.0;
            }
        }

        let config = DiagonalRidgeConfig {
            neuron_index: 10,
            max_lag: 8,
            min_diagonal_ratio: 1.3,
            min_diagonal_score: 0.2,
        };

        let err = verify_constructive_interference_surface_from_grid(&surface, &config)
            .expect_err("expected random noise to fail diagonal verification");
        assert!(err.contains("diagonal/axis ratio") || err.contains("diagonal autocorrelation"));
    }

    #[test]
    fn snake_curve_verification_passes_on_valid_grokking_curve() {
        // Simulate typical grokking: train converges early, val plateaus then jumps
        let train_acc = vec![
            (1, 0.10),
            (10, 0.50),
            (100, 0.80),
            (500, 0.98),
            (1000, 0.99),
            (2000, 0.995),
        ];
        let val_acc = vec![
            (1, 0.01),
            (10, 0.009),
            (100, 0.01),
            (500, 0.011),
            (1000, 0.009),
            (1500, 0.01),
            (2000, 0.92),
            (3000, 0.98),
        ];

        let config = SnakeCurveConfig {
            early_step_max: 1000,
            train_convergence_threshold: 0.95,
            val_plateau_min_step: 1500,
            val_plateau_max_acc: 0.05,
            val_final_threshold: 0.90,
        };

        let report = verify_snake_curve_shape(&train_acc, &val_acc, &config)
            .expect("expected snake curve verification to pass");

        assert_eq!(report.train_converged_by_step, 500);
        assert!(report.val_plateau_max_acc < config.val_plateau_max_acc);
        assert_eq!(report.val_grok_step, 2000);
        assert!(report.curve_shape_valid);
    }

    #[test]
    fn snake_curve_verification_fails_on_simultaneous_convergence() {
        // Both train and val converge at the same time (no grokking)
        let train_acc = vec![
            (1, 0.10),
            (100, 0.98),
            (500, 0.99),
        ];
        let val_acc = vec![
            (1, 0.10),
            (100, 0.97),
            (500, 0.98),
        ];

        let config = SnakeCurveConfig {
            early_step_max: 1000,
            train_convergence_threshold: 0.95,
            val_plateau_min_step: 200,
            val_plateau_max_acc: 0.05,
            val_final_threshold: 0.90,
        };

        let err = verify_snake_curve_shape(&train_acc, &val_acc, &config)
            .expect_err("expected snake curve verification to fail");
        assert!(err.contains("plateau region too high"));
    }

    #[test]
    fn snake_curve_verification_fails_on_slow_train_convergence() {
        // Train takes too long to converge (not the typical snake curve)
        let train_acc = vec![
            (1, 0.10),
            (500, 0.50),
            (1000, 0.70),
            (2000, 0.98),
        ];
        let val_acc = vec![
            (1, 0.01),
            (500, 0.01),
            (1000, 0.01),
            (2000, 0.92),
        ];

        let config = SnakeCurveConfig {
            early_step_max: 1000,
            train_convergence_threshold: 0.95,
            val_plateau_min_step: 1500,
            val_plateau_max_acc: 0.05,
            val_final_threshold: 0.90,
        };

        let err = verify_snake_curve_shape(&train_acc, &val_acc, &config)
            .expect_err("expected snake curve verification to fail");
        assert!(err.contains("converges at step 2000"));
    }

    #[test]
    fn snake_curve_verification_fails_without_val_jump() {
        // Validation never jumps up (no generalization)
        let train_acc = vec![
            (1, 0.10),
            (100, 0.98),
            (500, 0.99),
        ];
        let val_acc = vec![
            (1, 0.01),
            (100, 0.01),
            (500, 0.01),
            (2000, 0.02),
        ];

        let config = SnakeCurveConfig {
            early_step_max: 1000,
            train_convergence_threshold: 0.95,
            val_plateau_min_step: 200,
            val_plateau_max_acc: 0.05,
            val_final_threshold: 0.90,
        };

        let err = verify_snake_curve_shape(&train_acc, &val_acc, &config)
            .expect_err("expected snake curve verification to fail");
        assert!(err.contains("validation accuracy never reaches"));
    }

    #[test]
    fn snake_curve_verification_fails_on_wrong_phase_ordering() {
        // Val jumps before train converges (backwards snake)
        let train_acc = vec![
            (1, 0.10),
            (500, 0.50),
            (2000, 0.98),
        ];
        let val_acc = vec![
            (1, 0.01),
            (100, 0.95),
            (500, 0.98),
        ];

        let config = SnakeCurveConfig {
            early_step_max: 1000,
            train_convergence_threshold: 0.95,
            val_plateau_min_step: 200,
            val_plateau_max_acc: 0.05,
            val_final_threshold: 0.90,
        };

        let result = verify_snake_curve_shape(&train_acc, &val_acc, &config);
        // Should fail because val jumps before train converges (not a proper snake curve)
        // This can fail for multiple reasons, but should definitely fail
        assert!(result.is_err(), "Expected snake curve verification to fail but it passed");
    }

    #[test]
    fn weight_norm_decay_verification_passes_on_decreasing_norms() {
        let weight_norms = vec![
            (0, 100.0),
            (500, 85.0),
            (1000, 72.0),
            (1500, 63.0),
            (2000, 58.0),
            (2500, 55.0),
        ];

        let config = WeightNormDecayConfig {
            min_relative_decrease: 0.1,
            plateau_min_step: 1000,
            grok_step_window: 500,
        };

        let report = verify_weight_norm_decay(&weight_norms, Some(2000), &config)
            .expect("expected weight norm decay verification to pass");

        assert_eq!(report.initial_norm, 100.0);
        assert_eq!(report.final_norm, 55.0);
        assert!((report.relative_decrease - 0.45).abs() < 0.01);
        assert!(report.grok_step_norm.is_some());
    }

    #[test]
    fn weight_norm_decay_verification_fails_on_flat_norms() {
        let weight_norms = vec![
            (0, 100.0),
            (500, 99.0),
            (1000, 98.5),
            (1500, 98.0),
            (2000, 97.5),
        ];

        let config = WeightNormDecayConfig {
            min_relative_decrease: 0.1,
            plateau_min_step: 1000,
            grok_step_window: 500,
        };

        let err = verify_weight_norm_decay(&weight_norms, None, &config)
            .expect_err("expected weight norm decay verification to fail");
        assert!(err.contains("weight norm decrease"));
    }

    #[test]
    fn weight_norm_decay_verification_fails_on_increasing_norms() {
        let weight_norms = vec![
            (0, 100.0),
            (500, 95.0),
            (1000, 105.0), // Increase here
            (1500, 90.0),
            (2000, 85.0),
        ];

        let config = WeightNormDecayConfig {
            min_relative_decrease: 0.1,
            plateau_min_step: 1000,
            grok_step_window: 500,
        };

        let err = verify_weight_norm_decay(&weight_norms, None, &config)
            .expect_err("expected weight norm decay verification to fail");
        assert!(err.contains("weight norm increased"));
    }

    #[test]
    fn weight_norm_decay_verification_allows_small_noise() {
        // Weight decay with small numerical noise (within 1% tolerance)
        let weight_norms = vec![
            (0, 100.0),
            (500, 95.0),
            (1000, 90.5),  // Small increase within tolerance
            (1500, 90.0),
            (2000, 85.0),
        ];

        let config = WeightNormDecayConfig {
            min_relative_decrease: 0.1,
            plateau_min_step: 1000,
            grok_step_window: 500,
        };

        let report = verify_weight_norm_decay(&weight_norms, None, &config)
            .expect("expected weight norm decay with small noise to pass");

        assert_eq!(report.initial_norm, 100.0);
        assert_eq!(report.final_norm, 85.0);
        assert!((report.relative_decrease - 0.15).abs() < 0.01);
    }

    #[test]
    fn weight_norm_decay_verification_finds_grok_step_norm() {
        let weight_norms = vec![
            (0, 100.0),
            (1000, 80.0),
            (1800, 68.0),  // Within window of grok step (2200 - 500 = 1700)
            (2000, 65.0),
            (2200, 62.0),  // Around grok step
            (3000, 55.0),
        ];

        let config = WeightNormDecayConfig {
            min_relative_decrease: 0.1,
            plateau_min_step: 1000,
            grok_step_window: 500,
        };

        let report = verify_weight_norm_decay(&weight_norms, Some(2200), &config)
            .expect("expected weight norm decay verification to pass");

        assert!(report.grok_step_norm.is_some());
        let grok_norm = report.grok_step_norm.unwrap();
        // Should find the first norm in the window [1700, 2700], which is 68.0 at step 1800
        assert!((grok_norm - 68.0).abs() < 0.01);
    }
}
