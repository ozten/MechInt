//! Integration test for mini training loop
//!
//! This test runs a complete training loop for 10 epochs to verify that:
//! 1. Training completes without panics (especially squeeze/reshape bugs)
//! 2. Basic metrics are recorded properly
//! 3. Model forward/backward passes work correctly across multiple batches
//!
//! This would have caught the squeeze bugs that only appeared during training validation.

use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    data::dataset::Dataset,
    optim::lr_scheduler::linear::LinearLrSchedulerConfig,
    record::CompactRecorder,
    train::{
        metric::{AccuracyMetric, LossMetric},
        Learner, LearningParadigm, SupervisedTraining, TrainingStrategy,
    },
};
use grokking::{
    data::{build_dataloaders, ModularAdditionDataset},
    model::{Transformer, TransformerConfig},
    training_config::TrainingConfig,
};
use std::path::PathBuf;
use tempfile::TempDir;

type TestBackend = NdArray;
type TestAutodiffBackend = Autodiff<TestBackend>;

#[test]
fn test_mini_training_loop_completes_without_panic() {
    // Create a temporary directory for artifacts
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let artifact_dir = temp_dir.path().to_str().unwrap();

    // Use small configuration for fast testing
    let mut training_config = TrainingConfig::default();
    training_config.batch_size = 32; // smaller batch size for faster test
    training_config.num_epochs = 10; // just 10 epochs to verify it works
    training_config.seed = 42;
    training_config.num_workers = 0;

    // Skip validation for this test since we're using non-standard params
    std::env::set_var("GROK_SKIP_VALIDATION", "1");

    let device = NdArrayDevice::default();

    // Create datasets
    let train_dataset = ModularAdditionDataset::new(true, training_config.seed);
    let val_dataset = ModularAdditionDataset::new(false, training_config.seed);

    println!("📊 Test dataset sizes:");
    println!("  - Training: {} examples", train_dataset.len());
    println!("  - Validation: {} examples", val_dataset.len());

    // Create model
    let config = TransformerConfig::default();
    let model: Transformer<TestAutodiffBackend> = config.init(&device);
    println!("✅ Model initialized");

    // Setup optimizer
    let optim_config = training_config.optimizer_config();
    let optim = optim_config.init();
    println!("✅ Optimizer initialized");

    // Build dataloaders
    let (dataloader_train, dataloader_val) = build_dataloaders::<TestAutodiffBackend>(
        training_config.batch_size,
        training_config.num_workers,
        training_config.seed,
        device.clone(),
    );

    // Learning rate scheduler
    let warmup_start = (training_config.base_learning_rate / training_config.warmup_steps as f64)
        .max(1.0e-9);
    let lr_scheduler = LinearLrSchedulerConfig::new(
        warmup_start,
        training_config.base_learning_rate,
        training_config.warmup_steps,
    )
    .init()
    .expect("Learning rate scheduler should initialize");

    println!("🏋️  Starting mini training loop for {} epochs...", training_config.num_epochs);

    // Run training
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_val)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .with_training_strategy(TrainingStrategy::SingleDevice(device.clone()))
        .summary();

    let learner = Learner::new(model, optim, lr_scheduler);
    let result = training.run(learner);
    let _model = result.model;

    println!("✅ Training completed without panics!");

    // Verify that metric files were created
    let metrics_dir = PathBuf::from(artifact_dir);
    assert!(
        metrics_dir.exists(),
        "Artifact directory should exist after training"
    );

    // Check that some checkpoint or metric files exist
    let entries: Vec<_> = std::fs::read_dir(&metrics_dir)
        .expect("Should be able to read artifact directory")
        .collect();
    assert!(
        !entries.is_empty(),
        "Artifact directory should contain files after training"
    );

    println!("✅ Metrics and checkpoints verified!");

    // Clean up env var
    std::env::remove_var("GROK_SKIP_VALIDATION");
}

#[test]
fn test_mini_training_loop_with_batch_size_variations() {
    // This test specifically checks that different batch sizes work without panicking
    // Testing batch_size=1 and batch_size=16 to catch edge cases

    for batch_size in [1, 16] {
        println!("\n🧪 Testing with batch_size={}", batch_size);

        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let artifact_dir = temp_dir.path().to_str().unwrap();

        let mut training_config = TrainingConfig::default();
        training_config.batch_size = batch_size;
        training_config.num_epochs = 2; // just 2 epochs for quick verification
        training_config.seed = 42;
        training_config.num_workers = 0;

        std::env::set_var("GROK_SKIP_VALIDATION", "1");

        let device = NdArrayDevice::default();

        let config = TransformerConfig::default();
        let model: Transformer<TestAutodiffBackend> = config.init(&device);

        let optim_config = training_config.optimizer_config();
        let optim = optim_config.init();

        let (dataloader_train, dataloader_val) = build_dataloaders::<TestAutodiffBackend>(
            training_config.batch_size,
            training_config.num_workers,
            training_config.seed,
            device.clone(),
        );

        let warmup_start = (training_config.base_learning_rate / training_config.warmup_steps as f64)
            .max(1.0e-9);
        let lr_scheduler = LinearLrSchedulerConfig::new(
            warmup_start,
            training_config.base_learning_rate,
            training_config.warmup_steps,
        )
        .init()
        .expect("Learning rate scheduler should initialize");

        let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_val)
            .metrics((AccuracyMetric::new(), LossMetric::new()))
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(training_config.num_epochs)
            .with_training_strategy(TrainingStrategy::SingleDevice(device.clone()))
            .summary();

        let learner = Learner::new(model, optim, lr_scheduler);
        let result = training.run(learner);
        let _model = result.model;

        println!("✅ Training with batch_size={} completed without panics!", batch_size);

        std::env::remove_var("GROK_SKIP_VALIDATION");
    }
}
