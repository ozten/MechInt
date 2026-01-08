#![recursion_limit = "256"]

mod analysis;
mod checkpoint;
mod data;
mod export;
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
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();
    let video_mode = args.contains(&"--video".to_string());

    if video_mode {
        println!("📹 Video mode enabled: will generate frame-by-frame visualizations");
    }

    // Load config from environment variables (for parameter sweeps) or use defaults
    let training_config = TrainingConfig::from_env();
    training_config
        .validate_grokking_spec()
        .expect("Training config must match grokking spec");

    let weight_decay = match training_config.optimizer {
        training_config::OptimizerSpec::AdamW { weight_decay, .. } => weight_decay,
    };

    println!("🚀 Grokking experiment starting...");
    println!("Configuration:");
    println!("  - Model: 1-layer transformer (4 heads, dim=128, MLP=512)");
    println!("  - Optimizer: AdamW (β1=0.9, β2=0.98, weight decay={})", weight_decay);
    println!(
        "  - Learning rate: {} with {}-step linear warmup",
        training_config.base_learning_rate, training_config.warmup_steps
    );
    println!("  - Batch size: {}", training_config.batch_size);
    println!("  - Training epochs: {}", training_config.num_epochs);
    println!("  - Seed: {}", training_config.seed);
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
    println!("✅ AdamW optimizer initialized (β1=0.9, β2=0.98, weight decay={})", weight_decay);
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

    // Save labeled checkpoints at key epochs for grokking analysis
    println!();
    println!("{}", "=".repeat(80));
    println!("💾 Saving Key Checkpoints");
    println!("{}", "=".repeat(80));
    println!();

    // Epoch 100 (~step 1,000): End of memorization phase
    if num_epochs >= 100 {
        if let Err(e) = copy_checkpoint_set(artifact_dir, 100, "memorization_e100") {
            eprintln!("⚠️  Warning: Could not save memorization checkpoint (epoch 100): {}", e);
        } else {
            println!("💾 Saved memorization checkpoint at epoch 100 (~step 1,000)");
        }
    }

    // Epoch 500 (~step 5,000): Deep in plateau phase
    if num_epochs >= 500 {
        if let Err(e) = copy_checkpoint_set(artifact_dir, 500, "plateau_e500") {
            eprintln!("⚠️  Warning: Could not save plateau checkpoint (epoch 500): {}", e);
        } else {
            println!("💾 Saved plateau checkpoint at epoch 500 (~step 5,000)");
        }
    }

    // Epoch 1500 (~step 15,000): Post-grok solidification
    if num_epochs >= 1500 {
        if let Err(e) = copy_checkpoint_set(artifact_dir, 1500, "postgrok_e1500") {
            eprintln!("⚠️  Warning: Could not save post-grok checkpoint (epoch 1500): {}", e);
        } else {
            println!("💾 Saved post-grok checkpoint at epoch 1500 (~step 15,000)");
        }
    }

    println!();

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

    // Detect grokking transition with advanced analysis
    println!();
    println!("{}", "=".repeat(80));
    println!("🔍 Detecting Grokking Phase Transition");
    println!("{}", "=".repeat(80));
    println!();

    let grokking_detection = detect_grokking_transition(artifact_dir, num_epochs, steps_per_epoch);
    let grokking_epoch = if let Some((epoch, step, info)) = &grokking_detection {
        println!("🎯 GROKKING DETECTED!");
        println!("   Transition occurred at:");
        println!("     - Epoch: {}", epoch);
        println!("     - Step: {}", step);
        println!("   Validation accuracy metrics:");
        println!("     - Plateau baseline: {:.2}%", info.baseline_val_acc);
        println!("     - Post-grok accuracy: {:.2}%", info.grok_val_acc);
        println!("     - Accuracy jump: {:.2}% (within {} steps)",
                 info.accuracy_jump, info.window_size);
        println!();
        println!("   This sudden jump from ~random ({:.1}%) to ~perfect ({:.1}%)",
                 info.baseline_val_acc, info.grok_val_acc);
        println!("   is the hallmark of the grokking phenomenon!");
        println!();

        if let Err(e) = copy_checkpoint_set(artifact_dir, *epoch, "grokking") {
            eprintln!("⚠️  Warning: Could not save grokking checkpoint: {}", e);
        } else {
            println!("💾 Saved grokking checkpoint at epoch {} (step {})", epoch, step);
        }

        Some(*epoch)
    } else {
        println!("⚠️  No clear grokking transition detected.");
        println!("   Possible reasons:");
        println!("     - Training hasn't run long enough (expected ~step 7,000)");
        println!("     - Gradual improvement instead of sudden spike");
        println!("     - Grokking may occur beyond current epoch count");
        println!();
        println!("   Falling back to simple threshold detection...");

        let fallback = find_grokking_epoch(artifact_dir, num_epochs, steps_per_epoch, 90.0);
        if let Some(epoch) = fallback {
            println!("   Found validation accuracy >90% at epoch {}", epoch);
            if let Err(e) = copy_checkpoint_set(artifact_dir, epoch, "grokking") {
                eprintln!("⚠️  Warning: Could not save grokking checkpoint: {}", e);
            }
        } else {
            println!("   No epoch reached 90% validation accuracy yet.");
        }

        fallback
    };

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
    println!("{}", "=".repeat(80));
    println!("🔬 Weight Norm Evolution (Weight Decay Mechanism)");
    println!("{}", "=".repeat(80));
    println!();

    let mut weight_norm_history = analysis::WeightNormHistory::new();

    println!("🔍 Computing weight norms at key checkpoints...");
    println!("   (Weight decay drives the transition from high-norm memorization to low-norm generalization)");
    println!();

    // Initial model (step 0)
    let initial_norm = analysis::compute_model_weight_norm(&initial_model);
    weight_norm_history.add_snapshot(0, initial_norm);
    println!("   Step 0 - Weight norm: {:.4}", initial_norm);

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
                let norm = analysis::compute_model_weight_norm(&checkpoint_model);
                weight_norm_history.add_snapshot(checkpoint_step, norm);
                println!("   Step {} (epoch {}) - Weight norm: {:.4}", checkpoint_step, epoch, norm);
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
                    let norm = analysis::compute_model_weight_norm(&grokking_model);
                    weight_norm_history.add_snapshot(grokking_step, norm);
                    println!("   Step {} (GROK) - Weight norm: {:.4}", grokking_step, norm);
                }
                Err(err) => {
                    eprintln!("⚠️  Warning: Could not load grokking checkpoint: {}", err);
                }
            }
        } else {
            eprintln!("⚠️  Warning: Grokking checkpoint not found at {}", grokking_path);
        }
    }

    // Final model
    let final_norm = analysis::compute_model_weight_norm(&model);
    weight_norm_history.add_snapshot(total_steps, final_norm);
    println!("   Step {} (FINAL) - Weight norm: {:.4}", total_steps, final_norm);

    // Save weight norm history
    if !weight_norm_history.snapshots.is_empty() {
        let weight_norm_history_path = format!("{artifact_dir}/weight_norm_history.json");
        if let Err(e) = weight_norm_history.save(&weight_norm_history_path) {
            eprintln!("⚠️  Warning: Could not save weight norm history: {}", e);
        } else {
            println!("💾 Saved weight norm history to {}", weight_norm_history_path);
        }

        // Verify weight norm decay (should decrease during plateau phase)
        if weight_norm_history.snapshots.len() >= 3 {
            let decay_config = verify::WeightNormDecayConfig::default_for_grokking();

            // Convert Vec<WeightNormSnapshot> to Vec<(usize, f64)>
            let norm_tuples: Vec<(usize, f64)> = weight_norm_history
                .snapshots
                .iter()
                .map(|s| (s.step, s.total_norm))
                .collect();

            // Get grokking step if available
            let grok_step = grokking_epoch.map(|e| (e - 1) * steps_per_epoch);

            match verify::verify_weight_norm_decay(&norm_tuples, grok_step, &decay_config) {
                Ok(report) => {
                    println!(
                        "✅ Weight norm decay verified: {:.4} → {:.4} ({:.1}% decrease)",
                        report.initial_norm,
                        report.final_norm,
                        (report.initial_norm - report.final_norm) / report.initial_norm * 100.0
                    );
                    println!("   ⚡ Weight decay successfully drove the transition to generalizing circuit!");
                }
                Err(err) => {
                    eprintln!("ℹ️  Weight norm decay verification: {}", err);
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
    println!("📊 Generating Visualizations");
    println!("{}", "=".repeat(80));
    println!();

    // Create visualization directory
    let viz_dir = "artifacts/visualizations";
    std::fs::create_dir_all(viz_dir).ok();

    println!("📈 Core Grokking Plots:");
    println!();

    // Plot 1: Loss evolution (train and val)
    if let Err(e) = plotting::plot_loss_history_dual(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &format!("{}/loss_evolution.png", viz_dir),
    ) {
        eprintln!("⚠️  Warning: Could not generate loss plot: {}", e);
    }

    // Plot 2: Accuracy evolution (combined train/val)
    if let Err(e) = plotting::plot_accuracy_history(
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        &format!("{}/accuracy_evolution.png", viz_dir),
    ) {
        eprintln!("⚠️  Warning: Could not generate accuracy plot: {}", e);
    }

    // Plot 3: Combined grokking plot (accuracy + loss)
    if let Err(e) = plotting::plot_grokking_combined(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        &format!("{}/grokking_combined.png", viz_dir),
    ) {
        eprintln!("⚠️  Warning: Could not generate combined plot: {}", e);
    }

    println!();
    println!("📊 Log-Scale Snake Curve Plots:");
    println!();

    // Plot 3b: Loss evolution with LOG SCALE (for paper Figure 1)
    if let Err(e) = plotting::plot_loss_history_dual_logscale(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &format!("{}/loss_evolution_logscale.png", viz_dir),
    ) {
        eprintln!("⚠️  Warning: Could not generate log-scale loss plot: {}", e);
    }

    // Plot 3c: Accuracy evolution with LOG SCALE (for paper Figure 4)
    if let Err(e) = plotting::plot_accuracy_history_logscale(
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        &format!("{}/accuracy_evolution_logscale.png", viz_dir),
    ) {
        eprintln!("⚠️  Warning: Could not generate log-scale accuracy plot: {}", e);
    }

    // Plot 3d: Combined grokking plot with LOG SCALE
    if let Err(e) = plotting::plot_grokking_combined_logscale(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        &format!("{}/grokking_combined_logscale.png", viz_dir),
    ) {
        eprintln!("⚠️  Warning: Could not generate log-scale combined plot: {}", e);
    }

    println!();
    println!("🔬 FFT Analysis Plots:");
    println!();

    // Plot 4: FFT frequency distribution
    let fft_freq_data: Vec<(usize, usize, f64)> = fft_analysis
        .dominant_frequencies
        .iter()
        .map(|(token_id, freq)| (*token_id, *freq as usize, 0.0))
        .collect();

    if let Err(e) =
        plotting::plot_fft_frequency_distribution(&fft_freq_data, &format!("{}/fft_frequencies.png", viz_dir))
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

    if let Err(e) = plotting::plot_fft_spectra(&token_spectra, &format!("{}/fft_spectra.png", viz_dir)) {
        eprintln!("⚠️  Warning: Could not generate FFT spectra plot: {}", e);
    }

    println!();
    println!("🎨 Embedding Visualizations:");
    println!();

    // Generate embedding grid visualizations for key checkpoints
    generate_checkpoint_embeddings_visualization(
        artifact_dir,
        &device,
        grokking_epoch,
        num_epochs,
        viz_dir,
    );

    println!();

    println!();
    println!("📊 3D Activation Surface Data Export:");
    println!();

    // Export activation surfaces for Python visualization
    export_activation_surfaces(
        artifact_dir,
        &device,
        grokking_epoch,
        num_epochs,
    );

    println!();

    // Generate video frames if --video flag was set
    if video_mode {
        println!();
        println!("📹 Video Frame Generation:");
        println!();
        generate_video_frames(
            artifact_dir,
            &device,
            grokking_epoch,
            num_epochs,
        );
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

/// Detect grokking phase transition with rolling window analysis
/// Returns (epoch, step, detection_info) when grokking is detected
fn detect_grokking_transition(
    artifact_dir: &str,
    num_epochs: usize,
    steps_per_epoch: usize,
) -> Option<(usize, usize, GrokkingDetectionInfo)> {
    const WINDOW_SIZE: usize = 500; // steps to look back for spike detection
    const MIN_ACCURACY_JUMP: f64 = 20.0; // minimum % jump to qualify as grokking
    const MIN_PLATEAU_STEP: usize = 1000; // must be past memorization phase
    const TARGET_VAL_ACC: f64 = 90.0; // target validation accuracy

    let mut val_acc_history: Vec<(usize, f64)> = Vec::new();

    for epoch in 1..=num_epochs {
        let values = read_metric_entries(artifact_dir, "valid", epoch, "Accuracy");
        for (idx, value) in values.into_iter().enumerate() {
            let step = (epoch - 1) * steps_per_epoch + idx;
            val_acc_history.push((step, value));

            // Only start checking after plateau phase
            if step < MIN_PLATEAU_STEP {
                continue;
            }

            // Check if we've hit target accuracy
            if value >= TARGET_VAL_ACC {
                // Look back to find baseline (average over window before this jump)
                let window_start = val_acc_history.len().saturating_sub(WINDOW_SIZE);
                let baseline_values: Vec<f64> = val_acc_history[window_start..val_acc_history.len().saturating_sub(50)]
                    .iter()
                    .map(|(_, v)| *v)
                    .collect();

                if baseline_values.is_empty() {
                    continue;
                }

                let baseline_acc = baseline_values.iter().sum::<f64>() / baseline_values.len() as f64;
                let accuracy_jump = value - baseline_acc;

                // Verify this is a true spike (not gradual improvement)
                if accuracy_jump >= MIN_ACCURACY_JUMP {
                    let info = GrokkingDetectionInfo {
                        baseline_val_acc: baseline_acc,
                        grok_val_acc: value,
                        accuracy_jump,
                        window_size: WINDOW_SIZE,
                    };
                    return Some((epoch, step, info));
                }
            }
        }
    }

    None
}

/// Information about detected grokking transition
#[derive(Debug, Clone)]
struct GrokkingDetectionInfo {
    baseline_val_acc: f64,
    grok_val_acc: f64,
    accuracy_jump: f64,
    window_size: usize,
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

/// Generate embedding visualizations for key checkpoints
fn generate_checkpoint_embeddings_visualization(
    artifact_dir: &str,
    device: &WgpuDevice,
    grokking_epoch: Option<usize>,
    num_epochs: usize,
    viz_dir: &str,
) {
    println!("   Generating embedding visualizations for key checkpoints...");

    // Define checkpoints to visualize
    let mut checkpoints = vec![
        (1, "initial"),
        (100, "memorization_e100"),
        (500, "plateau_e500"),
    ];

    // Add grokking checkpoint if detected
    if let Some(epoch) = grokking_epoch {
        checkpoints.push((epoch, "grokking"));
    }

    // Add final checkpoint
    checkpoints.push((num_epochs, "final"));

    for (epoch, label) in checkpoints {
        // Try labeled checkpoint first (for initial/grokking/final)
        let labeled_path = Path::new(artifact_dir)
            .join("checkpoint_labeled")
            .join(format!("model-{}.mpk", label));

        let checkpoint_path = if labeled_path.exists() {
            labeled_path
        } else {
            // Fall back to epoch-based checkpoint
            Path::new(artifact_dir)
                .join("checkpoint")
                .join(format!("model-{}.mpk", epoch))
        };

        if !checkpoint_path.exists() {
            println!("   ⚠️  Skipping {}: checkpoint not found", label);
            continue;
        }

        match checkpoint::load_checkpoint::<MyAutodiffBackend>(
            checkpoint_path.to_str().unwrap(),
            device,
        ) {
            Ok(model) => {
                // Extract embeddings
                let embeddings = analysis::extract_all_embeddings(&model);

                if embeddings.is_empty() || embeddings[0].is_empty() {
                    println!("   ⚠️  Skipping {}: no embeddings found", label);
                    continue;
                }

                // Select 3 interesting dimensions (high variance)
                let interesting_dims = plotting::select_interesting_dimensions(&embeddings, 3);

                if interesting_dims.len() < 3 {
                    println!("   ⚠️  Skipping {}: insufficient dimensions", label);
                    continue;
                }

                let dims: [usize; 3] = [
                    interesting_dims[0],
                    interesting_dims[1],
                    interesting_dims[2],
                ];

                // Generate 3x3 embedding grid
                let output_path = format!("{}/embeddings_3x3_{}.png", viz_dir, label);
                let title = format!("Embedding Grid - {} (epoch {})", label, epoch);

                match plotting::plot_embedding_grid_3x3(&embeddings, &dims, &output_path, &title) {
                    Ok(_) => {
                        println!("   ✅ Generated embedding grid for {}", label);
                    }
                    Err(e) => {
                        eprintln!("   ⚠️  Could not generate embedding grid for {}: {}", label, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("   ⚠️  Could not load checkpoint for {}: {}", label, e);
            }
        }
    }
}

/// Export activation surface data for post-grokking checkpoint
/// This data can be visualized using scripts/visualize_activation_surface.py
fn export_activation_surfaces(
    artifact_dir: &str,
    device: &WgpuDevice,
    grokking_epoch: Option<usize>,
    num_epochs: usize,
) {
    use crate::export::ActivationSurface;

    println!("   Exporting activation surfaces for 3D visualization...");

    // We'll export surfaces from post-grokking checkpoint (or final if no grokking detected)
    let checkpoint_label = if grokking_epoch.is_some() {
        "postgrok_e1500"
    } else {
        "final"
    };

    let checkpoint_epoch = if let Some(_epoch) = grokking_epoch {
        // Try to use post-grokking checkpoint at epoch 1500
        if num_epochs >= 1500 {
            1500
        } else {
            num_epochs
        }
    } else {
        num_epochs
    };

    // Try labeled checkpoint first
    let labeled_path = Path::new(artifact_dir)
        .join("checkpoint_labeled")
        .join(format!("model-{}.mpk", checkpoint_label));

    let checkpoint_path = if labeled_path.exists() {
        labeled_path
    } else {
        // Fall back to epoch-based checkpoint
        Path::new(artifact_dir)
            .join("checkpoint")
            .join(format!("model-{}.mpk", checkpoint_epoch))
    };

    if !checkpoint_path.exists() {
        println!("   ⚠️  No post-grokking checkpoint found, skipping activation surface export");
        return;
    }

    match checkpoint::load_checkpoint::<MyAutodiffBackend>(
        checkpoint_path.to_str().unwrap(),
        device,
    ) {
        Ok(model) => {
            let modulus = ModularAdditionDataset::modulus();

            // Export surfaces for several neurons (indices 0, 10, 20, 30, ..., up to 5 neurons)
            let d_ff = 512; // From config
            let neuron_indices: Vec<usize> = (0..5).map(|i| i * (d_ff / 5).min(50)).collect();

            for &neuron_index in &neuron_indices {
                if neuron_index >= d_ff {
                    continue;
                }

                match verify::collect_mlp_activation_surface(&model, device, neuron_index) {
                    Ok(surface) => {
                        let surface_data = ActivationSurface::new(neuron_index, modulus, surface);

                        let output_path = format!(
                            "{}/activation_surface_neuron_{}.json",
                            artifact_dir, neuron_index
                        );

                        match surface_data.save_json(&output_path) {
                            Ok(_) => {
                                println!(
                                    "   ✅ Exported activation surface for neuron {} to {}",
                                    neuron_index, output_path
                                );
                            }
                            Err(e) => {
                                eprintln!(
                                    "   ⚠️  Failed to save activation surface for neuron {}: {}",
                                    neuron_index, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "   ⚠️  Failed to collect activation surface for neuron {}: {}",
                            neuron_index, e
                        );
                    }
                }
            }

            println!(
                "   💡 Visualize surfaces with: python scripts/visualize_activation_surface.py {}/activation_surface_neuron_0.json",
                artifact_dir
            );
        }
        Err(e) => {
            eprintln!(
                "   ⚠️  Could not load checkpoint for activation surface export: {}",
                e
            );
        }
    }
}

/// Generate video frames showing embedding evolution throughout training
///
/// Creates a sequence of 7x7 embedding grid visualizations at regular intervals,
/// suitable for creating an animation of the grokking phenomenon.
fn generate_video_frames(
    artifact_dir: &str,
    device: &WgpuDevice,
    grokking_epoch: Option<usize>,
    num_epochs: usize,
) {
    println!("   Generating video frames for embedding evolution...");

    let video_dir = Path::new(artifact_dir).join("video_frames");
    if let Err(e) = fs::create_dir_all(&video_dir) {
        eprintln!("   ⚠️  Could not create video_frames directory: {}", e);
        return;
    }

    // Define frame schedule: dense early, sparse later
    // This captures the rapid changes during memorization and grokking
    let mut frame_epochs = vec![1, 10, 25, 50, 75, 100]; // Early phase (0-100)

    // Memorization phase (100-500)
    for e in (150..=500).step_by(50) {
        if e <= num_epochs {
            frame_epochs.push(e);
        }
    }

    // Plateau and grokking phase (500-1000)
    for e in (550..=1000).step_by(50) {
        if e <= num_epochs {
            frame_epochs.push(e);
        }
    }

    // Post-grokking phase (1000+)
    for e in (1100..=num_epochs).step_by(100) {
        frame_epochs.push(e);
    }

    // Always include the final epoch
    if !frame_epochs.contains(&num_epochs) {
        frame_epochs.push(num_epochs);
    }

    println!("   📹 Will generate {} frames", frame_epochs.len());

    let mut successful_frames = 0;

    for (frame_idx, epoch) in frame_epochs.iter().enumerate() {
        // Try labeled checkpoint first (for special epochs)
        let labeled_checkpoints = vec![
            (1, "initial"),
            (100, "memorization_e100"),
            (500, "plateau_e500"),
            (1500, "postgrok_e1500"),
            (num_epochs, "final"),
        ];

        let grok_epoch = grokking_epoch.unwrap_or(0);
        let mut labeled_path = None;

        // Check if this epoch matches a labeled checkpoint
        for (label_epoch, label) in labeled_checkpoints {
            if *epoch == label_epoch {
                let path = Path::new(artifact_dir)
                    .join("checkpoint_labeled")
                    .join(format!("model-{}.mpk", label));
                if path.exists() {
                    labeled_path = Some(path);
                    break;
                }
            }
        }

        // Check for grokking checkpoint
        if grokking_epoch.is_some() && *epoch == grok_epoch {
            let path = Path::new(artifact_dir)
                .join("checkpoint_labeled")
                .join("model-grokking.mpk");
            if path.exists() {
                labeled_path = Some(path);
            }
        }

        // Fall back to epoch-based checkpoint
        let checkpoint_path = labeled_path.unwrap_or_else(|| {
            Path::new(artifact_dir)
                .join("checkpoint")
                .join(format!("model-{}.mpk", epoch))
        });

        if !checkpoint_path.exists() {
            // Skip missing checkpoints (expected for many epochs)
            continue;
        }

        // Load checkpoint and extract embeddings
        match checkpoint::load_checkpoint::<MyAutodiffBackend>(
            checkpoint_path.to_str().unwrap(),
            device,
        ) {
            Ok(model) => {
                let embeddings = analysis::extract_all_embeddings(&model);

                if embeddings.is_empty() || embeddings[0].is_empty() {
                    eprintln!("   ⚠️  Skipping epoch {}: no embeddings found", epoch);
                    continue;
                }

                // Select 7 interesting dimensions (high variance)
                let interesting_dims = plotting::select_interesting_dimensions(&embeddings, 7);

                if interesting_dims.len() < 7 {
                    eprintln!("   ⚠️  Skipping epoch {}: insufficient dimensions (need 7, got {})",
                             epoch, interesting_dims.len());
                    continue;
                }

                let dims: [usize; 7] = [
                    interesting_dims[0],
                    interesting_dims[1],
                    interesting_dims[2],
                    interesting_dims[3],
                    interesting_dims[4],
                    interesting_dims[5],
                    interesting_dims[6],
                ];

                // Generate 7x7 embedding grid with zero-padded frame number
                let output_path = format!(
                    "{}/frame_{:04}_epoch_{:05}.png",
                    video_dir.display(),
                    frame_idx,
                    epoch
                );

                let title = format!("Embedding Evolution - Epoch {}", epoch);

                match plotting::plot_embedding_grid_fast(&embeddings, &dims, &output_path, &title) {
                    Ok(_) => {
                        successful_frames += 1;
                        if successful_frames % 10 == 0 {
                            println!("   ✅ Generated {} frames...", successful_frames);
                        }
                    }
                    Err(e) => {
                        eprintln!("   ⚠️  Could not generate frame for epoch {}: {}", epoch, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("   ⚠️  Could not load checkpoint for epoch {}: {}", epoch, e);
            }
        }
    }

    println!("   ✅ Generated {} video frames in {}",
             successful_frames, video_dir.display());
    println!("   💡 Create video with: ffmpeg -framerate 10 -pattern_type glob -i '{}/frame_*.png' -c:v libx264 -pix_fmt yuv420p grokking_evolution.mp4",
             video_dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create synthetic accuracy data for testing
    fn create_test_accuracy_data(
        dir: &Path,
        epochs: Vec<(usize, Vec<f64>)>,
    ) -> std::io::Result<()> {
        for (epoch, values) in epochs {
            let epoch_dir = dir.join("valid").join(format!("epoch-{}", epoch));
            fs::create_dir_all(&epoch_dir)?;

            let accuracy_file = epoch_dir.join("Accuracy.log");
            let mut file = fs::File::create(accuracy_file)?;

            for value in values {
                // Write in NumericEntry format: "value,count"
                writeln!(file, "{},100", value)?;
            }
        }
        Ok(())
    }

    #[test]
    fn test_detect_grokking_transition_finds_spike() {
        let temp_dir = TempDir::new().unwrap();
        let artifact_dir = temp_dir.path();

        // Simulate grokking:
        // - Epochs 1-150: plateau at ~1% (random guessing)
        // - Epoch 151: sudden jump to 95% (grokking at step ~1510)
        let steps_per_epoch = 10;

        let mut epochs = vec![];

        // Plateau phase (epochs 1-150) - exceeds MIN_PLATEAU_STEP=1000
        for epoch in 1..=150 {
            epochs.push((epoch, vec![1.0, 1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.1, 0.9, 1.0]));
        }

        // Grokking epoch (151): sudden spike at step ~1510
        epochs.push((151, vec![2.0, 5.0, 15.0, 35.0, 60.0, 80.0, 92.0, 95.0, 96.0, 97.0]));

        // Post-grok (epochs 152-155)
        for epoch in 152..=155 {
            epochs.push((epoch, vec![96.5, 97.0, 97.5, 98.0, 97.8, 98.2, 98.0, 97.9, 98.1, 98.0]));
        }

        create_test_accuracy_data(artifact_dir, epochs).unwrap();

        // Run detection
        let result = detect_grokking_transition(
            artifact_dir.to_str().unwrap(),
            155,
            steps_per_epoch,
        );

        // Debug: print what we got
        if let Some((epoch, step, ref info)) = result {
            eprintln!("Detected: epoch={}, step={}, baseline={:.2}, grok={:.2}, jump={:.2}",
                     epoch, step, info.baseline_val_acc, info.grok_val_acc, info.accuracy_jump);
        } else {
            eprintln!("No grokking detected!");
        }

        assert!(result.is_some(), "Should detect grokking transition");
        let (epoch, step, info) = result.unwrap();

        assert_eq!(epoch, 151, "Should detect grokking at epoch 151");
        assert!(step >= 1500 && step <= 1510, "Step should be around 1500-1510, got {}", step);
        assert!(info.baseline_val_acc < 5.0, "Baseline should be near random (~1%), got {:.2}", info.baseline_val_acc);
        assert!(info.grok_val_acc >= 90.0, "Post-grok accuracy should be >90%, got {:.2}", info.grok_val_acc);
        assert!(info.accuracy_jump >= 20.0, "Accuracy jump should be >20%, got {:.2}", info.accuracy_jump);
    }

    #[test]
    fn test_detect_grokking_transition_no_spike() {
        let temp_dir = TempDir::new().unwrap();
        let artifact_dir = temp_dir.path();

        // Simulate gradual improvement (no grokking)
        let steps_per_epoch = 10;
        let mut epochs = vec![];

        for epoch in 1..=100 {
            let base_acc = epoch as f64 * 0.8; // gradual linear improvement
            epochs.push((epoch, vec![base_acc; 10]));
        }

        create_test_accuracy_data(artifact_dir, epochs).unwrap();

        // Run detection
        let result = detect_grokking_transition(
            artifact_dir.to_str().unwrap(),
            100,
            steps_per_epoch,
        );

        assert!(result.is_none(), "Should NOT detect grokking with gradual improvement");
    }

    #[test]
    fn test_detect_grokking_transition_early_spike_ignored() {
        let temp_dir = TempDir::new().unwrap();
        let artifact_dir = temp_dir.path();

        // Simulate early spike (before MIN_PLATEAU_STEP=1000)
        let steps_per_epoch = 10;
        let mut epochs = vec![];

        // Early spike at epoch 10 (step ~100) - should be ignored
        for epoch in 1..=10 {
            epochs.push((epoch, vec![1.0; 10]));
        }
        epochs.push((11, vec![95.0; 10])); // spike at step ~110

        for epoch in 12..=50 {
            epochs.push((epoch, vec![96.0; 10]));
        }

        create_test_accuracy_data(artifact_dir, epochs).unwrap();

        // Run detection
        let result = detect_grokking_transition(
            artifact_dir.to_str().unwrap(),
            50,
            steps_per_epoch,
        );

        // Should not detect because spike is before MIN_PLATEAU_STEP (1000)
        assert!(result.is_none(), "Should ignore spike before step 1000");
    }
}
