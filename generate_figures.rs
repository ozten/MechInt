/// Generate paper figures from saved training data
/// Usage: cargo run --bin generate_figures --release
use grokking::{analysis, plotting};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Generating paper figures from saved data...");
    println!();

    // Load loss history
    let loss_data = fs::read_to_string("artifacts/loss_history.json")?;
    let loss_history: analysis::LossHistory = serde_json::from_str(&loss_data)?;

    // Load accuracy history
    let acc_data = fs::read_to_string("artifacts/accuracy_history.json")?;
    let accuracy_history: analysis::AccuracyHistory = serde_json::from_str(&acc_data)?;

    println!("📁 Loaded training histories:");
    println!("  - Loss snapshots: {}", loss_history.train_snapshots.len());
    println!("  - Accuracy snapshots: {}", accuracy_history.train_snapshots.len());
    println!();

    // Create figures directory
    fs::create_dir_all("figures")?;

    // Generate Figure 1: Log-scale accuracy plot
    println!("🎨 Generating Figure 1: Accuracy evolution (log scale)...");
    plotting::plot_accuracy_history_logscale(
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "figures/figure1_accuracy_logscale.png",
    )?;

    // Generate Figure 4: Log-scale loss double-descent plot
    println!("🎨 Generating Figure 4: Loss double-descent (log scale)...");
    plotting::plot_loss_history_dual_logscale(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        "figures/figure4_loss_logscale.png",
    )?;

    // Generate combined figure with log scale
    println!("🎨 Generating combined grokking figure (log scale)...");
    plotting::plot_grokking_combined_logscale(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "figures/grokking_combined_logscale.png",
    )?;

    // Also generate linear scale versions for comparison
    println!("🎨 Generating linear scale versions for comparison...");
    plotting::plot_accuracy_history(
        &accuracy_history.train_snapshots,
        &accuracy_history.val_snapshots,
        "figures/accuracy_linear.png",
    )?;

    plotting::plot_loss_history_dual(
        &loss_history.train_snapshots,
        &loss_history.val_snapshots,
        "figures/loss_linear.png",
    )?;

    println!();
    println!("✅ All figures generated successfully!");
    println!();
    println!("📂 Output files:");
    println!("  figures/figure1_accuracy_logscale.png    - Paper Figure 1 reproduction");
    println!("  figures/figure4_loss_logscale.png        - Paper Figure 4 reproduction");
    println!("  figures/grokking_combined_logscale.png   - Combined view (log scale)");
    println!("  figures/accuracy_linear.png              - Accuracy (linear scale)");
    println!("  figures/loss_linear.png                  - Loss (linear scale)");
    println!();

    Ok(())
}
