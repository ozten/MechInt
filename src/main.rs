mod analysis;
mod checkpoint;
mod data;
mod model;
mod plotting;
mod verify;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::Dataset,
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Int, Tensor},
};
use data::ModularAdditionDataset;
use model::{Transformer, TransformerConfig};

type Backend = Wgpu;
type MyAutodiffBackend = Autodiff<Backend>;

/// Compute learning rate with linear warmup (paper uses 10 steps)
fn get_learning_rate_with_warmup(step: usize, base_lr: f64, warmup_steps: usize) -> f64 {
    if step < warmup_steps {
        // Linear warmup from 0 to base_lr
        base_lr * (step as f64 + 1.0) / warmup_steps as f64
    } else {
        base_lr
    }
}

fn main() {
    println!("🚀 Grokking experiment starting...");
    println!("Configuration:");
    println!("  - Model: 2-layer transformer (4 heads, dim=128, MLP=512)");
    println!("  - Optimizer: Adam (β1=0.9, β2=0.98, no weight decay)");
    println!("  - Learning rate: 1e-3 with 10-step linear warmup");
    println!("  - Batch size: 512");
    println!("  - Training steps: 100,000 (logs every 500 steps)");
    println!("  - Target: Modular addition (a + b) mod 97");
    println!("  - Tracking: Train/Val Loss AND Accuracy (full datasets)");
    println!();

    // Setup device
    let device = WgpuDevice::default();
    println!("📱 Using device: {:?}", device);
    println!();

    // Create datasets
    let train_dataset = ModularAdditionDataset::new(true, 42);
    let val_dataset = ModularAdditionDataset::new(false, 42);
    println!("📊 Dataset sizes:");
    println!("  - Training: {} examples", train_dataset.len());
    println!("  - Validation: {} examples", val_dataset.len());
    println!();

    // Create model
    let config = TransformerConfig::default();
    let mut model: Transformer<MyAutodiffBackend> = config.init(&device);
    println!("✅ Model initialized");

    // Setup optimizer to match paper (Figure 1: Adam without weight decay, β2=0.98)
    let optim_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.98)  // Paper uses 0.98, not default 0.999
        .with_epsilon(1e-8);
    let mut optim = optim_config.init();
    println!("✅ Adam optimizer initialized (β1=0.9, β2=0.98, no weight decay)");
    println!();

    // Training configuration (matching paper)
    let batch_size = 512;
    let max_steps = 100_000; // Paper uses 1e5-1e6, running 100k (~1 hour)
    let log_interval = 500; // Log every 500 steps (200 logs total)
    let base_learning_rate = 1e-3;
    let warmup_steps = 10; // Paper uses linear warmup over 10 updates

    println!("🏋️  Starting training for {} steps...", max_steps);
    println!("{}", "=".repeat(80));
    println!();

    // Track loss over time (both train and val)
    let mut loss_history = analysis::LossHistory::new();

    // Track accuracy over time
    let mut accuracy_history = analysis::AccuracyHistory::new();

    // Track if we've saved grokking checkpoint
    let mut grokking_checkpoint_saved = false;

    // Training loop
    for step in 0..max_steps {
        // Sample a batch from training data
        let (inputs, targets) = sample_batch(&train_dataset, batch_size, &device);

        // Forward pass
        let logits = model.forward(inputs.clone());

        // Compute loss
        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits.clone(), targets.clone());

        // Backward pass
        let grads = loss.backward();

        // Update parameters with warmup learning rate
        let lr = get_learning_rate_with_warmup(step, base_learning_rate, warmup_steps);
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);

        // Logging
        if step % log_interval == 0 || step == max_steps - 1 {
            let train_acc = compute_accuracy(&model, &train_dataset, batch_size, &device);
            let val_acc = compute_accuracy(&model, &val_dataset, batch_size, &device);

            // Compute train and val loss on full datasets
            let train_loss = compute_loss(&model, &train_dataset, &device);
            let val_loss = compute_loss(&model, &val_dataset, &device);

            // Track loss evolution (both train and val)
            loss_history.add_snapshot(step, train_loss, val_loss);

            // Track accuracy
            accuracy_history.add_snapshot(step, train_acc, val_acc);

            // Save milestone checkpoints
            if step == 0 {
                let _ = checkpoint::save_labeled_checkpoint(&model, "step_0_initial");
            } else if step == 500 && train_acc > 0.99 {
                let _ = checkpoint::save_labeled_checkpoint(&model, "step_500_memorized");
            }

            println!(
                "Step {:6} | Train Loss: {:.4} | Val Loss: {:.4} | Train Acc: {:6.2}% | Val Acc: {:6.2}%",
                step,
                train_loss,
                val_loss,
                train_acc * 100.0,
                val_acc * 100.0
            );

            // Check for grokking
            if val_acc > 0.90 && step > 1000 && !grokking_checkpoint_saved {
                println!();
                println!("🎉 GROKKING DETECTED! Validation accuracy > 90%");
                println!("   Step: {}", step);
                println!("   Train Acc: {:.2}%", train_acc * 100.0);
                println!("   Val Acc: {:.2}%", val_acc * 100.0);

                // Save checkpoint when grokking is first detected
                if let Err(e) = checkpoint::save_labeled_checkpoint(&model, &format!("grokking_step_{}", step)) {
                    eprintln!("⚠️  Warning: Could not save grokking checkpoint: {}", e);
                } else {
                    grokking_checkpoint_saved = true;
                }
            }
        }
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("✅ Training complete!");
    println!();

    // CRITICAL: Verify on complete dataset (no sampling!)
    println!("{}", "=".repeat(80));
    println!("🔍 VERIFICATION (Full Dataset - No Sampling)");
    println!("{}", "=".repeat(80));
    println!();

    let (true_train_acc, true_val_acc) = verify::verify_full_accuracy(&model, &device);

    println!();
    verify::test_specific_examples(&model, &device);

    println!();
    if true_train_acc < 0.99 || true_val_acc < 0.99 {
        println!("⚠️  WARNING: Model did NOT actually learn the algorithm!");
        println!("   This confirms we were fooled by random sampling variance.");
    } else {
        println!("✅ CONFIRMED: Model actually learned modular addition!");
    }

    // Perform FFT analysis on learned embeddings
    println!();
    println!("{}", "=".repeat(80));
    println!("📊 Post-Training Analysis");
    println!("{}", "=".repeat(80));
    println!();

    let fft_analysis = analysis::analyze_embeddings_fft(&model);
    if let Err(e) = analysis::save_fft_analysis(&fft_analysis, "fft_analysis.json") {
        eprintln!("⚠️  Warning: Could not save FFT analysis: {}", e);
    }

    println!();
    println!("📉 Loss Evolution Analysis:");
    if let (Some(&(_, train_init)), Some(&(_, val_init))) =
        (loss_history.train_snapshots.first(), loss_history.val_snapshots.first())
    {
        if let (Some(&(_, train_final)), Some(&(_, val_final))) =
            (loss_history.train_snapshots.last(), loss_history.val_snapshots.last())
        {
            println!("   Initial - Train: {:.4}, Val: {:.4}", train_init, val_init);
            println!("   Final   - Train: {:.4}, Val: {:.4}", train_final, val_final);
            println!("   Train decrease: {:.4}", train_init - train_final);
            println!("   Val decrease: {:.4}", val_init - val_final);
        }
    }

    if let Err(e) = loss_history.save("loss_history.json") {
        eprintln!("⚠️  Warning: Could not save loss history: {}", e);
    }

    if let Err(e) = accuracy_history.save("accuracy_history.json") {
        eprintln!("⚠️  Warning: Could not save accuracy history: {}", e);
    }

    // Generate plots
    println!();
    println!("{}", "=".repeat(80));
    println!("📊 Generating Plots");
    println!("{}", "=".repeat(80));
    println!();

    // Plot 1: Loss evolution (train and val)
    if let Err(e) = plotting::plot_loss_history_dual(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        "plots/loss_evolution.png",
    ) {
        eprintln!("⚠️  Warning: Could not generate loss plot: {}", e);
    }

    // Plot 2: Accuracy evolution (combined train/val)
    if let Err(e) = plotting::plot_accuracy_history(
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "plots/accuracy_evolution.png",
    ) {
        eprintln!("⚠️  Warning: Could not generate accuracy plot: {}", e);
    }

    // Plot 3: Combined grokking plot (accuracy + loss)
    if let Err(e) = plotting::plot_grokking_combined(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "plots/grokking_combined.png",
    ) {
        eprintln!("⚠️  Warning: Could not generate combined plot: {}", e);
    }

    // Plot 3b: Loss evolution with LOG SCALE (for paper Figure 1)
    if let Err(e) = plotting::plot_loss_history_dual_logscale(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        "plots/loss_evolution_logscale.png",
    ) {
        eprintln!("⚠️  Warning: Could not generate log-scale loss plot: {}", e);
    }

    // Plot 3c: Accuracy evolution with LOG SCALE (for paper Figure 4)
    if let Err(e) = plotting::plot_accuracy_history_logscale(
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "plots/accuracy_evolution_logscale.png",
    ) {
        eprintln!("⚠️  Warning: Could not generate log-scale accuracy plot: {}", e);
    }

    // Plot 3d: Combined grokking plot with LOG SCALE
    if let Err(e) = plotting::plot_grokking_combined_logscale(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "plots/grokking_combined_logscale.png",
    ) {
        eprintln!("⚠️  Warning: Could not generate log-scale combined plot: {}", e);
    }

    // Plot 4: FFT frequency distribution
    let fft_freq_data: Vec<(usize, usize, f64)> = fft_analysis
        .dominant_frequencies
        .iter()
        .map(|(token_id, freq)| (*token_id, *freq as usize, 0.0))
        .collect();

    if let Err(e) =
        plotting::plot_fft_frequency_distribution(&fft_freq_data, "plots/fft_frequencies.png")
    {
        eprintln!("⚠️  Warning: Could not generate FFT frequency plot: {}", e);
    }

    // Plot 5: FFT spectra for selected tokens (0, 10, 20, 48, 49, 96)
    let selected_tokens = vec![0, 10, 20, 48, 49, 96];
    let token_spectra: Vec<(usize, Vec<f64>)> = selected_tokens
        .iter()
        .filter_map(|&token_id| {
            fft_analysis
                .frequency_magnitudes
                .get(&token_id)
                .map(|spec| (token_id, spec.clone()))
        })
        .collect();

    if let Err(e) = plotting::plot_fft_spectra(&token_spectra, "plots/fft_spectra.png") {
        eprintln!("⚠️  Warning: Could not generate FFT spectra plot: {}", e);
    }

    // Save final model checkpoint
    println!();
    println!("{}", "=".repeat(80));
    println!("💾 Saving Model Checkpoints");
    println!("{}", "=".repeat(80));
    println!();

    if let Err(e) = checkpoint::save_labeled_checkpoint(&model, "final") {
        eprintln!("⚠️  Warning: Could not save final checkpoint: {}", e);
    }

    // List all saved checkpoints
    match checkpoint::list_checkpoints() {
        Ok(checkpoints) => {
            if !checkpoints.is_empty() {
                println!();
                println!("📦 Available checkpoints:");
                for ckpt in checkpoints {
                    println!("   - checkpoints/{}", ckpt);
                }
            }
        }
        Err(e) => {
            eprintln!("⚠️  Warning: Could not list checkpoints: {}", e);
        }
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("🎉 All analysis complete!");
    println!("{}", "=".repeat(80));
}

/// Sample a random batch from the dataset
fn sample_batch<B: AutodiffBackend>(
    dataset: &ModularAdditionDataset,
    batch_size: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    use rand::seq::SliceRandom;

    let indices: Vec<usize> = (0..dataset.len()).collect();
    let mut rng = rand::thread_rng();
    let batch_indices: Vec<usize> = indices
        .choose_multiple(&mut rng, batch_size.min(dataset.len()))
        .copied()
        .collect();

    let mut inputs_vec = Vec::new();
    let mut targets_vec = Vec::new();

    for idx in batch_indices {
        let (input, target) = dataset.get(idx).unwrap();
        inputs_vec.extend(input);
        targets_vec.push(target as i32);
    }

    let actual_batch_size = targets_vec.len();

    let inputs = Tensor::<B, 1, Int>::from_ints(inputs_vec.as_slice(), device)
        .reshape([actual_batch_size, 3]);
    let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

    (inputs, targets)
}

/// Compute loss on FULL dataset (no random sampling)
fn compute_loss<B: AutodiffBackend>(
    model: &Transformer<B>,
    dataset: &ModularAdditionDataset,
    device: &B::Device,
) -> f64 {
    let total = dataset.len();
    let mut total_loss = 0.0f64;
    let mut total_count = 0;

    // Process in batches to avoid memory issues
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
        let targets = Tensor::<B, 1, Int>::from_ints(targets_vec.as_slice(), device);

        // Forward pass WITHOUT gradients - use inner backend
        let logits = model.clone().valid().forward(inputs.inner());

        // Compute loss on inner backend
        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits, targets.inner());

        // Extract loss value as f64
        let loss_data = loss.into_data();
        let loss_vec: Vec<f32> = loss_data.to_vec().unwrap();
        let loss_val = loss_vec[0] as f64;
        total_loss += loss_val * batch_len as f64;
        total_count += batch_len;
    }

    total_loss / total_count as f64
}

/// Compute accuracy on FULL dataset (no random sampling)
/// This matches the paper's methodology and avoids variance from sampling
fn compute_accuracy<B: AutodiffBackend>(
    model: &Transformer<B>,
    dataset: &ModularAdditionDataset,
    _batch_size: usize,
    device: &B::Device,
) -> f32 {
    let total = dataset.len();
    let mut total_correct = 0;

    // Process in batches to avoid memory issues
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

        // Forward pass WITHOUT gradients
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
