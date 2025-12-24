#![recursion_limit = "256"]

mod analysis;
mod checkpoint;
mod data;
mod model;
mod plotting;
mod training;
mod verify;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::{lr_scheduler::linear::LinearLrSchedulerConfig, AdamConfig},
    record::CompactRecorder,
    train::{
        metric::{AccuracyMetric, LossMetric, NumericEntry},
        Learner, LearningParadigm, SupervisedTraining, TrainingStrategy,
    },
};
use data::{build_dataloaders, ModularAdditionDataset};
use model::{Transformer, TransformerConfig};
use std::{fs, path::Path};

type WgpuBackend = Wgpu;
type MyAutodiffBackend = Autodiff<WgpuBackend>;

fn main() {
    println!("🚀 Grokking experiment starting...");
    println!("Configuration:");
    println!("  - Model: 2-layer transformer (4 heads, dim=128, MLP=512)");
    println!("  - Optimizer: Adam (β1=0.9, β2=0.98, no weight decay)");
    println!("  - Learning rate: 1e-3 with 10-step linear warmup");
    println!("  - Batch size: 512");
    println!("  - Training steps: 100,000 (mapped to epochs)");
    println!("  - Target: Modular addition (a + b) mod 97");
    println!("  - Tracking: Train/Val Loss AND Accuracy (Burn metrics)");
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
    let model: Transformer<MyAutodiffBackend> = config.init(&device);
    println!("✅ Model initialized");

    // Setup optimizer to match paper (Figure 1: Adam without weight decay, β2=0.98)
    let optim_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.98)  // Paper uses 0.98, not default 0.999
        .with_epsilon(1e-8);
    let optim = optim_config.init();
    println!("✅ Adam optimizer initialized (β1=0.9, β2=0.98, no weight decay)");
    println!();

    // Training configuration (matching paper)
    let batch_size = 512;
    let num_workers = 0;
    let seed = 42;
    let max_steps = 100_000; // Paper uses 1e5-1e6, running 100k (~1 hour)
    let base_learning_rate = 1e-3;
    let warmup_steps = 10; // Paper uses linear warmup over 10 updates

    let steps_per_epoch = (train_dataset.len() + batch_size - 1) / batch_size;
    let num_epochs = (max_steps + steps_per_epoch - 1) / steps_per_epoch;

    let artifact_dir = "artifacts";
    std::fs::create_dir_all(artifact_dir).ok();

    // Build Burn dataloaders
    let (dataloader_train, dataloader_val) =
        build_dataloaders::<MyAutodiffBackend>(batch_size, num_workers, seed, device.clone());

    let warmup_start = (base_learning_rate / warmup_steps as f64).max(1.0e-9);
    let lr_scheduler = LinearLrSchedulerConfig::new(
        warmup_start,
        base_learning_rate,
        warmup_steps,
    )
    .init()
    .expect("Learning rate scheduler should initialize");

    println!(
        "🏋️  Starting training for {} epochs (~{} steps)...",
        num_epochs,
        num_epochs * steps_per_epoch
    );
    println!("{}", "=".repeat(80));
    println!();

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_val)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(num_epochs)
        .with_training_strategy(TrainingStrategy::SingleDevice(device.clone()))
        .summary();

    let learner = Learner::new(model, optim, lr_scheduler);
    let result = training.run(learner);
    let model = result.model;

    let (loss_history, accuracy_history) =
        load_metric_history(artifact_dir, num_epochs, steps_per_epoch);

    if let Some(epoch) = find_grokking_epoch(artifact_dir, num_epochs, steps_per_epoch, 90.0) {
        if let Err(e) = copy_checkpoint_set(artifact_dir, epoch, "grokking") {
            eprintln!("⚠️  Warning: Could not save grokking checkpoint: {}", e);
        }
    }

    if let Err(e) = copy_checkpoint_set(artifact_dir, num_epochs, "final") {
        eprintln!("⚠️  Warning: Could not save final checkpoint: {}", e);
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
    let fft_analysis_path = format!("{artifact_dir}/fft_analysis.json");
    if let Err(e) = analysis::save_fft_analysis(&fft_analysis, &fft_analysis_path) {
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

    let loss_history_path = format!("{artifact_dir}/loss_history.json");
    if let Err(e) = loss_history.save(&loss_history_path) {
        eprintln!("⚠️  Warning: Could not save loss history: {}", e);
    }

    let accuracy_history_path = format!("{artifact_dir}/accuracy_history.json");
    if let Err(e) = accuracy_history.save(&accuracy_history_path) {
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

    println!();
    println!("{}", "=".repeat(80));
    println!("🎉 All analysis complete!");
    println!("{}", "=".repeat(80));
}

fn load_metric_history(
    artifact_dir: &str,
    num_epochs: usize,
    steps_per_epoch: usize,
) -> (analysis::LossHistory, analysis::AccuracyHistory) {
    let mut loss_history = analysis::LossHistory::new();
    let mut accuracy_history = analysis::AccuracyHistory::new();

    for epoch in 1..=num_epochs {
        let train_loss = read_metric_entries(artifact_dir, "train", epoch, "Loss");
        for (idx, value) in train_loss.into_iter().enumerate() {
            let step = (epoch - 1) * steps_per_epoch + idx;
            loss_history.train_snapshots.push((step, value));
        }

        let val_loss = read_metric_entries(artifact_dir, "valid", epoch, "Loss");
        for (idx, value) in val_loss.into_iter().enumerate() {
            let step = (epoch - 1) * steps_per_epoch + idx;
            loss_history.val_snapshots.push((step, value));
        }

        let train_acc = read_metric_entries(artifact_dir, "train", epoch, "Accuracy");
        for (idx, value) in train_acc.into_iter().enumerate() {
            let step = (epoch - 1) * steps_per_epoch + idx;
            accuracy_history
                .train_snapshots
                .push((step, (value / 100.0) as f32));
        }

        let val_acc = read_metric_entries(artifact_dir, "valid", epoch, "Accuracy");
        for (idx, value) in val_acc.into_iter().enumerate() {
            let step = (epoch - 1) * steps_per_epoch + idx;
            accuracy_history
                .val_snapshots
                .push((step, (value / 100.0) as f32));
        }
    }

    (loss_history, accuracy_history)
}

fn read_metric_entries(
    artifact_dir: &str,
    split: &str,
    epoch: usize,
    metric_name: &str,
) -> Vec<f64> {
    let path = Path::new(artifact_dir)
        .join(split)
        .join(format!("epoch-{epoch}"))
        .join(format!("{metric_name}.log"));

    let Ok(contents) = fs::read_to_string(path) else {
        return Vec::new();
    };

    contents
        .lines()
        .filter_map(|line| NumericEntry::deserialize(line).ok())
        .map(|entry| entry.current())
        .collect()
}

fn find_grokking_epoch(
    artifact_dir: &str,
    num_epochs: usize,
    steps_per_epoch: usize,
    threshold_pct: f64,
) -> Option<usize> {
    for epoch in 1..=num_epochs {
        let values = read_metric_entries(artifact_dir, "valid", epoch, "Accuracy");
        for (idx, value) in values.into_iter().enumerate() {
            let step = (epoch - 1) * steps_per_epoch + idx;
            if step > 1000 && value >= threshold_pct {
                return Some(epoch);
            }
        }
    }

    None
}

fn copy_checkpoint_set(
    artifact_dir: &str,
    epoch: usize,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let source_dir = Path::new(artifact_dir).join("checkpoint");
    let dest_dir = Path::new(artifact_dir).join("checkpoint_labeled");
    fs::create_dir_all(&dest_dir)?;

    for name in ["model", "optim", "scheduler"] {
        let src = source_dir.join(format!("{name}-{epoch}.mpk"));
        if !src.exists() {
            continue;
        }

        let dst = dest_dir.join(format!("{name}-{label}.mpk"));
        fs::copy(src, dst)?;
    }

    Ok(())
}
