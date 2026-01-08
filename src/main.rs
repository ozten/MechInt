#![recursion_limit = "256"]

mod analysis;
mod checkpoint;
mod data;
mod model;
mod plotting;
mod training;
mod training_config;
mod verify;

use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    data::dataset::Dataset,
    optim::lr_scheduler::linear::LinearLrSchedulerConfig,
    record::CompactRecorder,
    train::{
        metric::{AccuracyMetric, LossMetric, NumericEntry},
        Learner, LearningParadigm, SupervisedTraining, TrainingStrategy,
    },
};
use data::{build_dataloaders, ModularAdditionDataset};
use model::{Transformer, TransformerConfig};
use training_config::TrainingConfig;
use std::{fs, path::Path};

type WgpuBackend = Wgpu;
type MyAutodiffBackend = Autodiff<WgpuBackend>;

fn main() {
    let training_config = TrainingConfig::default();
    training_config
        .validate_grokking_spec()
        .expect("Training config must match grokking spec");

    println!("🚀 Grokking experiment starting...");
    println!("Configuration:");
    println!("  - Model: 1-layer transformer (4 heads, dim=128, MLP=512)");
    println!("  - Optimizer: AdamW (β1=0.9, β2=0.98, weight decay=1.0)");
    println!(
        "  - Learning rate: {} with {}-step linear warmup",
        training_config.base_learning_rate, training_config.warmup_steps
    );
    println!("  - Batch size: {}", training_config.batch_size);
    println!("  - Training epochs: {}", training_config.num_epochs);
    println!(
        "  - Target: Modular addition (a + b) mod {}",
        ModularAdditionDataset::modulus()
    );
    println!("  - Tracking: Train/Val Loss AND Accuracy (Burn metrics)");
    println!();

    // Setup device
    let device = WgpuDevice::default();
    println!("📱 Using device: {:?}", device);
    println!();

    // Create datasets
    let train_dataset = ModularAdditionDataset::new(true, training_config.seed);
    let val_dataset = ModularAdditionDataset::new(false, training_config.seed);
    println!("📊 Dataset sizes:");
    println!("  - Training: {} examples", train_dataset.len());
    println!("  - Validation: {} examples", val_dataset.len());
    println!();

    // Create model
    let config = TransformerConfig::default();
    let model: Transformer<MyAutodiffBackend> = config.init(&device);
    println!("✅ Model initialized");

    // Setup optimizer to match paper (AdamW with high weight decay, β2=0.98)
    let optim_config = training_config.optimizer_config();
    let optim = optim_config.init();
    println!("✅ AdamW optimizer initialized (β1=0.9, β2=0.98, weight decay=1.0)");
    println!();

    // Training configuration (matching paper)
    let batch_size = training_config.batch_size;
    let num_workers = training_config.num_workers;
    let seed = training_config.seed;
    let num_epochs = training_config.num_epochs;
    let base_learning_rate = training_config.base_learning_rate;
    let warmup_steps = training_config.warmup_steps;

    let steps_per_epoch = training_config.steps_per_epoch(train_dataset.len());
    let total_steps = training_config.total_steps(train_dataset.len());

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
        total_steps
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

    let grokking_config =
        verify::GrokkingVerificationConfig::default_for_modulus(ModularAdditionDataset::modulus());
    match verify::verify_grokking_phase_transition(
        &loss_history,
        &accuracy_history,
        &grokking_config,
    ) {
        Ok(report) => {
            println!(
                "✅ Grokking phase transition detected (train>99% at step {}, val>99% at step {})",
                report.early_train_step, report.transition_step
            );
        }
        Err(err) => {
            eprintln!("⚠️  Warning: Grokking phase verification failed: {}", err);
        }
    }

    let grokking_epoch = find_grokking_epoch(artifact_dir, num_epochs, steps_per_epoch, 90.0);
    if let Some(epoch) = grokking_epoch {
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

    println!();
    println!("{}", "=".repeat(80));
    println!("🔬 Advanced Metrics (Restricted/Excluded Loss)");
    println!("{}", "=".repeat(80));
    println!();

    let restricted_top_k = 5usize;
    let excluded_top_k = 5usize;
    let metrics_batch_size = training_config.batch_size;
    let mut restricted_history = analysis::RestrictedLossHistory::new();
    let mut excluded_history = analysis::ExcludedLossHistory::new();

    // Key checkpoints based on expected timeline:
    // - Epoch 100 (~step 1,000): End of memorization
    // - Epoch 500 (~step 5,000): Deep in plateau
    // - Grokking epoch (~step 7,000): Phase transition
    // - Final epoch: Post-grok solidification
    let checkpoint_epochs = vec![100, 500];

    println!("🔍 Computing restricted/excluded loss at key checkpoints...");
    println!("   (These metrics detect internal restructuring before grokking)");
    println!();

    // Initial model (step 0)
    let initial_model: Transformer<MyAutodiffBackend> = TransformerConfig::default().init(&device);
    match analysis::compute_restricted_loss(
        &initial_model,
        &train_dataset,
        &device,
        restricted_top_k,
        metrics_batch_size,
    ) {
        Ok(loss) => {
            restricted_history.add_snapshot(0, loss);
            println!("   Step 0 - Restricted loss (top_k={}): {:.4}", restricted_top_k, loss);
        }
        Err(err) => {
            eprintln!("⚠️  Warning: Could not compute restricted loss (initial): {}", err);
        }
    }

    match analysis::compute_excluded_loss(
        &initial_model,
        &train_dataset,
        &device,
        excluded_top_k,
        metrics_batch_size,
    ) {
        Ok(loss) => {
            excluded_history.add_snapshot(0, loss);
            println!("   Step 0 - Excluded loss (top_k={}): {:.4}", excluded_top_k, loss);
        }
        Err(err) => {
            eprintln!("⚠️  Warning: Could not compute excluded loss (initial): {}", err);
        }
    }

    // Checkpoint epochs (100, 500)
    for &epoch in &checkpoint_epochs {
        if epoch > num_epochs {
            continue;
        }

        let checkpoint_step = (epoch - 1) * steps_per_epoch;
        let checkpoint_path = format!("{artifact_dir}/checkpoint/model-{epoch}.mpk");

        if !Path::new(&checkpoint_path).exists() {
            continue;
        }

        match checkpoint::load_checkpoint::<MyAutodiffBackend>(&checkpoint_path, &device) {
            Ok(checkpoint_model) => {
                match analysis::compute_restricted_loss(
                    &checkpoint_model,
                    &train_dataset,
                    &device,
                    restricted_top_k,
                    metrics_batch_size,
                ) {
                    Ok(loss) => {
                        restricted_history.add_snapshot(checkpoint_step, loss);
                        println!("   Step {} (epoch {}) - Restricted loss: {:.4}",
                                 checkpoint_step, epoch, loss);
                    }
                    Err(err) => {
                        eprintln!("⚠️  Warning: Could not compute restricted loss (epoch {}): {}",
                                  epoch, err);
                    }
                }

                match analysis::compute_excluded_loss(
                    &checkpoint_model,
                    &train_dataset,
                    &device,
                    excluded_top_k,
                    metrics_batch_size,
                ) {
                    Ok(loss) => {
                        excluded_history.add_snapshot(checkpoint_step, loss);
                        println!("   Step {} (epoch {}) - Excluded loss: {:.4}",
                                 checkpoint_step, epoch, loss);
                    }
                    Err(err) => {
                        eprintln!("⚠️  Warning: Could not compute excluded loss (epoch {}): {}",
                                  epoch, err);
                    }
                }
            }
            Err(err) => {
                eprintln!("⚠️  Warning: Could not load checkpoint at epoch {}: {}", epoch, err);
            }
        }
    }

    // Grokking checkpoint (if detected)
    if let Some(epoch) = grokking_epoch {
        let grokking_step = (epoch - 1) * steps_per_epoch;
        let grokking_path = format!("{artifact_dir}/checkpoint_labeled/model-grokking.mpk");
        if Path::new(&grokking_path).exists() {
            match checkpoint::load_checkpoint::<MyAutodiffBackend>(&grokking_path, &device) {
                Ok(grokking_model) => {
                    match analysis::compute_restricted_loss(
                        &grokking_model,
                        &train_dataset,
                        &device,
                        restricted_top_k,
                        metrics_batch_size,
                    ) {
                        Ok(loss) => {
                            restricted_history.add_snapshot(grokking_step, loss);
                            println!("   Step {} (GROK) - Restricted loss: {:.4}",
                                     grokking_step, loss);
                        }
                        Err(err) => {
                            eprintln!(
                                "⚠️  Warning: Could not compute restricted loss (grokking): {}",
                                err
                            );
                        }
                    }

                    match analysis::compute_excluded_loss(
                        &grokking_model,
                        &train_dataset,
                        &device,
                        excluded_top_k,
                        metrics_batch_size,
                    ) {
                        Ok(loss) => {
                            excluded_history.add_snapshot(grokking_step, loss);
                            println!("   Step {} (GROK) - Excluded loss: {:.4}",
                                     grokking_step, loss);
                        }
                        Err(err) => {
                            eprintln!(
                                "⚠️  Warning: Could not compute excluded loss (grokking): {}",
                                err
                            );
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "⚠️  Warning: Could not load grokking checkpoint: {}",
                        err
                    );
                }
            }
        } else {
            eprintln!(
                "⚠️  Warning: Grokking checkpoint not found at {}",
                grokking_path
            );
        }
    }

    // Final model
    match analysis::compute_restricted_loss(
        &model,
        &train_dataset,
        &device,
        restricted_top_k,
        metrics_batch_size,
    ) {
        Ok(loss) => {
            restricted_history.add_snapshot(total_steps, loss);
            println!("   Step {} (FINAL) - Restricted loss: {:.4}", total_steps, loss);
        }
        Err(err) => {
            eprintln!("⚠️  Warning: Could not compute restricted loss (final): {}", err);
        }
    }

    match analysis::compute_excluded_loss(
        &model,
        &train_dataset,
        &device,
        excluded_top_k,
        metrics_batch_size,
    ) {
        Ok(loss) => {
            excluded_history.add_snapshot(total_steps, loss);
            println!("   Step {} (FINAL) - Excluded loss: {:.4}", total_steps, loss);
        }
        Err(err) => {
            eprintln!("⚠️  Warning: Could not compute excluded loss (final): {}", err);
        }
    }

    // Save restricted loss history
    if !restricted_history.snapshots.is_empty() {
        let restricted_history_path = format!("{artifact_dir}/restricted_loss_history.json");
        if let Err(e) = restricted_history.save(&restricted_history_path) {
            eprintln!("⚠️  Warning: Could not save restricted loss history: {}", e);
        } else {
            println!("💾 Saved restricted loss history to {}", restricted_history_path);
        }

        // Verify restricted loss drop (should drop BEFORE val accuracy improves)
        if restricted_history.snapshots.len() >= 3 {
            let drop_config = verify::RestrictedLossDropConfig {
                min_relative_decrease: 0.3,
            };
            match verify::verify_restricted_loss_drop(&restricted_history.snapshots, &drop_config) {
                Ok(report) => {
                    println!(
                        "✅ Restricted loss drop detected at step {} (baseline {:.4}, dropped to {:.4})",
                        report.drop_step, report.baseline_loss, report.drop_loss
                    );
                    println!("   ⚡ This internal restructuring preceded the grokking transition!");
                }
                Err(err) => {
                    eprintln!("ℹ️  Restricted loss drop: {}", err);
                }
            }
        }
    }

    // Save excluded loss history
    if !excluded_history.snapshots.is_empty() {
        let excluded_history_path = format!("{artifact_dir}/excluded_loss_history.json");
        if let Err(e) = excluded_history.save(&excluded_history_path) {
            eprintln!("⚠️  Warning: Could not save excluded loss history: {}", e);
        } else {
            println!("💾 Saved excluded loss history to {}", excluded_history_path);
        }

        // Verify excluded loss spike (should spike as grokking approaches)
        if excluded_history.snapshots.len() >= 3 {
            let spike_config = verify::ExcludedLossSpikeConfig {
                min_relative_increase: 0.5,
            };
            match verify::verify_excluded_loss_spike(&excluded_history.snapshots, &spike_config) {
                Ok(report) => {
                    println!(
                        "✅ Excluded loss spike detected at step {} (baseline {:.4}, peak {:.4})",
                        report.spike_step, report.baseline_loss, report.spike_loss
                    );
                    println!("   ⚡ This spike indicates the model switching from memorization to Fourier algorithm!");
                }
                Err(err) => {
                    eprintln!("ℹ️  Excluded loss spike: {}", err);
                }
            }
        }
    }

    println!();

    let pre_checkpoint = Path::new(artifact_dir).join("checkpoint").join("model-1.mpk");
    let post_checkpoint_grokking = Path::new(artifact_dir)
        .join("checkpoint_labeled")
        .join("model-grokking.mpk");
    let post_checkpoint_final = Path::new(artifact_dir)
        .join("checkpoint_labeled")
        .join("model-final.mpk");

    if pre_checkpoint.exists()
        && (post_checkpoint_grokking.exists() || post_checkpoint_final.exists())
    {
        let post_path = if post_checkpoint_grokking.exists() {
            post_checkpoint_grokking
        } else {
            post_checkpoint_final
        };

        match (
            checkpoint::load_checkpoint::<MyAutodiffBackend>(
                pre_checkpoint.to_str().unwrap(),
                &device,
            ),
            checkpoint::load_checkpoint::<MyAutodiffBackend>(post_path.to_str().unwrap(), &device),
        ) {
            (Ok(pre_model), Ok(post_model)) => {
                let manifold_config = verify::PairwiseManifoldConfig::default_for_modulus(
                    ModularAdditionDataset::modulus(),
                );
                match verify::verify_mlp_pairwise_manifold_transition(
                    &pre_model,
                    &post_model,
                    &device,
                    &manifold_config,
                ) {
                    Ok(report) => {
                        println!(
                            "✅ Pairwise manifold rings detected (pre avg {:.3}, post avg {:.3}, pairs {}/{})",
                            report.pre_average_score,
                            report.post_average_score,
                            report.passing_pairs,
                            report.total_pairs
                        );
                    }
                    Err(err) => {
                        eprintln!(
                            "⚠️  Warning: Pairwise manifold verification failed: {}",
                            err
                        );
                    }
                }
            }
            (Err(err), _) => {
                eprintln!(
                    "⚠️  Warning: Could not load pre-grok checkpoint {}: {}",
                    pre_checkpoint.display(),
                    err
                );
            }
            (_, Err(err)) => {
                eprintln!(
                    "⚠️  Warning: Could not load post-grok checkpoint {}: {}",
                    post_path.display(),
                    err
                );
            }
        }
    } else {
        eprintln!(
            "⚠️  Warning: Pairwise manifold verification skipped (missing checkpoints)"
        );
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
