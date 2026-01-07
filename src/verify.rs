use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Int, Tensor},
};

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
}
