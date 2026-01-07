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
}
