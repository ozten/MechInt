use burn::{
    data::dataset::Dataset,
    module::AutodiffModule,
    tensor::{backend::AutodiffBackend, Int, Tensor},
};

use crate::data::ModularAdditionDataset;
use crate::model::Transformer;

/// Verify that the model actually learned modular addition
/// by testing it on ALL possible examples systematically
pub fn verify_full_accuracy<B: AutodiffBackend>(
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
fn test_all_examples<B: AutodiffBackend>(
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

        let logits = model.clone().valid().forward(inputs.inner());
        let predictions = logits.argmax(1).squeeze::<1>(1);
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

/// Test specific examples to see if model actually computes mod 97
pub fn test_specific_examples<B: AutodiffBackend>(
    model: &Transformer<B>,
    device: &B::Device,
) {
    println!();
    println!("🧪 Testing specific examples:");

    let test_cases = vec![
        (0, 0, 0),      // 0 + 0 = 0
        (1, 1, 2),      // 1 + 1 = 2
        (50, 50, 3),    // 50 + 50 = 100 mod 97 = 3
        (96, 1, 0),     // 96 + 1 = 97 mod 97 = 0
        (48, 49, 0),    // 48 + 49 = 97 mod 97 = 0
        (96, 96, 95),   // 96 + 96 = 192 mod 97 = 95
        (10, 20, 30),   // 10 + 20 = 30
        (60, 60, 23),   // 60 + 60 = 120 mod 97 = 23
    ];

    for (a, b, expected) in test_cases {
        let input_vec = vec![a as i32, b as i32, 97]; // 97 is the '=' token
        let input = Tensor::<B, 1, Int>::from_ints(input_vec.as_slice(), device)
            .reshape([1, 3]);

        let logits = model.clone().valid().forward(input.inner());
        let prediction = logits.argmax(1).squeeze::<1>(1);
        let pred_value: i32 = prediction.into_data().to_vec().unwrap()[0];

        let correct = if pred_value == expected as i32 { "✓" } else { "✗" };
        println!("   {} + {} mod 97 = {} | Model: {} {}",
            a, b, expected, pred_value, correct);
    }
}
