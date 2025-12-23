use burn::{
    module::{AutodiffModule, Module},
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use std::path::Path;

use crate::model::Transformer;

/// Save model checkpoint to disk
pub fn save_checkpoint<B: AutodiffBackend>(
    model: &Transformer<B>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Convert model to record (without autodiff wrapper)
    let record = model.clone().valid().into_record();

    // Save using CompactRecorder (binary format)
    CompactRecorder::new()
        .record(record, path.into())
        .map_err(|e| format!("Failed to save checkpoint: {:?}", e))?;

    println!("💾 Model checkpoint saved to: {}", path);
    Ok(())
}

/// Load model checkpoint from disk
pub fn load_checkpoint<B: AutodiffBackend>(
    path: &str,
    device: &B::Device,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    use crate::model::TransformerConfig;

    // Create a new model with default config
    let config = TransformerConfig::default();
    let model = config.init(device);

    // Load the record
    let record = CompactRecorder::new()
        .load(path.into(), device)
        .map_err(|e| format!("Failed to load checkpoint: {:?}", e))?;

    // Load record into model
    let model = model.load_record(record);

    println!("📂 Model checkpoint loaded from: {}", path);
    Ok(model)
}

/// Save a labeled checkpoint (e.g., "grokking", "final", "step_1000")
pub fn save_labeled_checkpoint<B: AutodiffBackend>(
    model: &Transformer<B>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("checkpoints/model_{}", label);
    save_checkpoint(model, &path)
}

/// List available checkpoints
pub fn list_checkpoints() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let checkpoint_dir = Path::new("checkpoints");

    if !checkpoint_dir.exists() {
        return Ok(Vec::new());
    }

    let mut checkpoints = Vec::new();

    for entry in std::fs::read_dir(checkpoint_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("mpk") {
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                checkpoints.push(filename.to_string());
            }
        }
    }

    checkpoints.sort();
    Ok(checkpoints)
}
