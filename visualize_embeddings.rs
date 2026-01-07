/// Visualize embeddings from model checkpoints
/// Usage: cargo run --bin visualize_embeddings --release
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use grokking::{analysis, checkpoint, plotting};
use std::path::{Path, PathBuf};

type Backend = Wgpu;
type MyAutodiffBackend = Autodiff<Backend>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Generating embedding visualizations from checkpoints...");
    println!();

    let device = WgpuDevice::default();

    // Create embeddings directory
    std::fs::create_dir_all("embeddings")?;

    let checkpoint_dir = Path::new("artifacts/checkpoint");
    let checkpoints = list_model_checkpoints(checkpoint_dir);

    if checkpoints.is_empty() {
        eprintln!("⚠️  No model checkpoints found in {}", checkpoint_dir.display());
        return Ok(());
    }

    for (epoch, path) in checkpoints {
        let title = format!("Checkpoint (Epoch {})", epoch);
        let label = format!("epoch_{}", epoch);
        println!("📂 Loading checkpoint: {}", path.display());

        match checkpoint::load_checkpoint::<MyAutodiffBackend>(path.to_str().unwrap(), &device) {
            Ok(model) => {
                println!("   ✓ Checkpoint loaded");

                // Extract all embeddings
                println!("   🔍 Extracting embeddings...");
                let embeddings = analysis::extract_all_embeddings(&model);
                println!("      Embeddings shape: [{}, {}]", embeddings.len(), embeddings[0].len());

                // Select 7 interesting dimensions based on variance
                println!("   📊 Selecting interesting dimensions...");
                let dimensions_vec = plotting::select_interesting_dimensions(&embeddings, 7);
                let dimensions: [usize; 7] = [
                    dimensions_vec[0],
                    dimensions_vec[1],
                    dimensions_vec[2],
                    dimensions_vec[3],
                    dimensions_vec[4],
                    dimensions_vec[5],
                    dimensions_vec[6],
                ];
                println!("      Selected dimensions: {:?}", dimensions);

                // Generate 7×7 grid visualization
                let output_path = format!("embeddings/embedding_grid_{}.png", label);
                println!("   🎨 Generating 7×7 grid...");
                plotting::plot_embedding_grid_fast(&embeddings, &dimensions, &output_path, &title)?;

                println!("   ✅ Visualization complete!");
                println!();
            }
            Err(e) => {
                eprintln!("   ⚠️  Could not load checkpoint: {}", e);
                eprintln!("   Skipping...");
                println!();
            }
        }
    }

    println!("================================================================================");
    println!("✅ All embedding visualizations generated!");
    println!("================================================================================");
    println!();
    println!("📂 Output files:");
    println!("   embeddings/embedding_grid_epoch_*.png     - Available checkpoints");
    println!();
    println!("💡 Look for:");
    println!("   - Sinusoidal patterns in first column (dimension vs token index)");
    println!("   - Circular patterns in scatter plots (sin/cos pairs)");
    println!("   - Progression from random → structured as training progresses");
    println!();

    Ok(())
}

fn list_model_checkpoints(checkpoint_dir: &Path) -> Vec<(usize, PathBuf)> {
    let mut checkpoints = Vec::new();
    let Ok(entries) = std::fs::read_dir(checkpoint_dir) else {
        return checkpoints;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(name) => name,
            None => continue,
        };

        if !file_name.starts_with("model-") || !file_name.ends_with(".mpk") {
            continue;
        }

        let epoch_str = &file_name["model-".len()..file_name.len() - ".mpk".len()];
        let Ok(epoch) = epoch_str.parse::<usize>() else {
            continue;
        };

        let mut base = path.clone();
        base.set_extension("");
        checkpoints.push((epoch, base));
    }

    checkpoints.sort_by_key(|(epoch, _)| *epoch);
    checkpoints
}
