/// Visualize embeddings from model checkpoints
/// Usage: cargo run --bin visualize_embeddings --release
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use grokking::{analysis, checkpoint, plotting};

type Backend = Wgpu;
type MyAutodiffBackend = Autodiff<Backend>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Generating embedding visualizations from checkpoints...");
    println!();

    let device = WgpuDevice::default();

    // Create embeddings directory
    std::fs::create_dir_all("embeddings")?;

    // List of checkpoints to visualize
    let checkpoints = vec![
        ("checkpoints/model_step_0_initial", "Initial (Random)", "initial"),
        ("checkpoints/model_step_500_memorized", "Memorized (Step 500)", "memorized"),
        ("checkpoints/model_grokking_step_2000", "Grokking (Step 2000)", "grokking"),
        ("checkpoints/model_final", "Final (Step 100k)", "final"),
    ];

    for (path, title, label) in checkpoints {
        println!("📂 Loading checkpoint: {}", path);

        match checkpoint::load_checkpoint::<MyAutodiffBackend>(path, &device) {
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

                // Generate 7×7 grid visualization (FAST version)
                let output_path = format!("embeddings/embedding_grid_{}.png", label);
                println!("   🎨 Generating 7×7 grid (fast pixel rendering)...");
                plotting::plot_embedding_grid_fast(&embeddings, &dimensions, &output_path, title)?;

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
    println!("   embeddings/embedding_grid_initial.png     - Random initialization");
    println!("   embeddings/embedding_grid_memorized.png   - After memorization");
    println!("   embeddings/embedding_grid_grokking.png    - During grokking transition");
    println!("   embeddings/embedding_grid_final.png       - Final learned embeddings");
    println!();
    println!("💡 Look for:");
    println!("   - Sinusoidal patterns in first column (dimension vs token index)");
    println!("   - Circular patterns in scatter plots (sin/cos pairs)");
    println!("   - Progression from random → structured as training progresses");
    println!();

    Ok(())
}
