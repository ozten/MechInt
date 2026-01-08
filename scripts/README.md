# Grokking Visualization and Analysis Scripts

This directory contains tools for visualizing and analyzing the grokking phenomenon in modular arithmetic learning, including:

- **3D activation surface visualization** - Visualize MLP neuron activation patterns
- **Weight decay sweep experiments** - Validate grokking robustness across different hyperparameters

## Setup

Install required Python dependencies:

```bash
pip install -r scripts/requirements.txt
```

Or install individually:

```bash
pip install numpy plotly
```

## Usage

### 1. Export Activation Data

The training pipeline automatically exports activation surface data during post-training analysis. These JSON files are saved to the artifacts directory:

```
artifacts/activation_surface_neuron_0.json
artifacts/activation_surface_neuron_102.json
artifacts/activation_surface_neuron_204.json
...
```

Each JSON file contains a 113x113 grid of activation values for all possible (x, y) input pairs.

### 2. Generate 3D Visualizations

Run the visualization script on any exported JSON file:

```bash
python scripts/visualize_activation_surface.py artifacts/activation_surface_neuron_0.json
```

This generates two HTML files:
- `activation_surface_neuron_0.html` - Interactive 3D surface plot
- `activation_surface_neuron_0_combined.html` - Combined view with 3D surface + 2D heatmap

Open the HTML files in your browser to interact with the 3D plots (rotate, zoom, pan).

### 3. Custom Output Path

Specify a custom output path:

```bash
python scripts/visualize_activation_surface.py artifacts/activation_surface_neuron_10.json my_plot.html
```

## What to Look For

The characteristic "corrugated iron" diagonal wave structure indicates the model has discovered the Fourier algorithm for modular addition:

- **Diagonal ridges**: Periodic waves along x+y and x-y diagonals
- **High periodicity strength**: Strong dominant frequencies in FFT analysis
- **Structured patterns**: Clear geometric structure vs random noise

These patterns emerge suddenly during grokking and persist in the post-grokking phase.

## Technical Details

The visualization script:
1. Loads the activation surface JSON data
2. Creates interactive Plotly 3D surface plots
3. Analyzes diagonal structure using FFT
4. Reports periodicity metrics

The activation data is collected from the post-ReLU MLP activations at the final token position (the "=" token) for all 12,769 possible input pairs (x, y) where 0 ≤ x, y < 113.

## References

See `docs/NOTES.md` Section 5.4 and Appendix A for the theoretical background on constructive interference and Fourier structure discovery.

---

# Weight Decay Sweep Experiments

This section describes how to run systematic weight decay sweep experiments to validate the robustness of the grokking phenomenon.

## Overview

Weight decay is the most critical hyperparameter for grokking. These experiments systematically vary weight decay values across multiple random seeds to validate:

1. **Baseline (WD=1.0)**: Should grok reliably by epoch ~7k-10k
2. **Lower WD (WD=0.5)**: Should delay or prevent grokking
3. **Higher WD (WD=2.0)**: Should accelerate grokking to ~5k epochs

## Quick Start

### 1. Run the Sweep

**Full experiment (90-135 GPU hours):**
```bash
./scripts/sweep_weight_decay.sh
```

**Quick test (for debugging, ~1-2 hours):**
```bash
./scripts/sweep_weight_decay.sh --quick
```

**Dry run (test without executing):**
```bash
./scripts/sweep_weight_decay.sh --dry-run
```

### 2. Analyze Results

After the sweep completes, analyze the results:

```bash
python scripts/analyze_weight_decay_sweep.py artifacts/sweep_weight_decay_YYYYMMDD_HHMMSS
```

This generates:
- Summary statistics for each weight decay value
- Grokking epoch distributions
- Success/failure rates
- Hypothesis validation results
- `analysis_summary.json` with detailed metrics

### 3. Generate Comparison Plots

Create interactive comparison visualizations:

```bash
python scripts/plot_sweep_comparison.py artifacts/sweep_weight_decay_YYYYMMDD_HHMMSS
```

This generates HTML plots in `artifacts/sweep_weight_decay_YYYYMMDD_HHMMSS/plots/`:
- `validation_accuracy_comparison.html` - Overlay of all runs' validation accuracy curves
- `grokking_epoch_distribution.html` - Box plot showing grokking epoch distributions
- `combined_comparison.html` - Multi-panel comparison view

## Environment Variables

The sweep script uses environment variables to control training parameters:

- `GROK_WEIGHT_DECAY`: Weight decay value (0.5, 1.0, 2.0)
- `GROK_SEED`: Random seed (42, 123, 456)
- `GROK_NUM_EPOCHS`: Number of training epochs (default: 10000)
- `GROK_SKIP_VALIDATION`: Set to "1" to skip strict parameter validation

You can also run individual experiments manually:

```bash
GROK_WEIGHT_DECAY=2.0 GROK_SEED=42 GROK_NUM_EPOCHS=10000 GROK_SKIP_VALIDATION=1 \
  cargo run --release --bin grokking
```

## Experiment Design

The default sweep runs 9 experiments in a 3×3 grid:

| Weight Decay | Seeds        | Expected Outcome              |
|--------------|--------------|-------------------------------|
| 0.5          | 42, 123, 456 | Delayed/failed grokking       |
| 1.0          | 42, 123, 456 | Grokking ~epoch 7k-10k        |
| 2.0          | 42, 123, 456 | Accelerated grokking ~epoch 5k|

Each run produces:
- Training logs (`training_output.log`)
- Loss/accuracy history JSON files
- Checkpoints at key phases
- Visualizations
- Grokking detection results

## Success Criteria

The experiment validates these hypotheses:

1. **✅ WD=1.0 groks by epoch 10k**: All baseline runs should reach 90%+ validation accuracy
2. **✅ WD=2.0 groks earlier than WD=1.0**: Higher weight decay accelerates grokking
3. **✅ WD=0.5 groks later or fails**: Lower weight decay delays or prevents grokking

## Compute Requirements

- **Full sweep**: ~90-135 GPU hours (9 runs × 10-15 hours each)
- **Quick sweep**: ~1-2 hours (100 epochs, for testing only)
- **Storage**: ~1-2 GB per run (checkpoints + artifacts)

## Output Structure

```
artifacts/sweep_weight_decay_YYYYMMDD_HHMMSS/
├── run_log.txt                           # Overall sweep log
├── analysis_summary.json                 # Analysis results
├── plots/                                # Generated visualizations
│   ├── validation_accuracy_comparison.html
│   ├── grokking_epoch_distribution.html
│   └── combined_comparison.html
├── wd0.5_seed42/                         # Individual run results
│   ├── training_output.log
│   ├── loss_history.json
│   ├── accuracy_history.json
│   ├── checkpoint_labeled/
│   └── visualizations/
├── wd0.5_seed123/
├── wd0.5_seed456/
├── wd1.0_seed42/
└── ... (9 run directories total)
```

## Troubleshooting

**Issue**: Sweep script fails with "cargo not found"
- **Solution**: Ensure Rust and Cargo are installed and in your PATH

**Issue**: Python scripts fail with import errors
- **Solution**: Install requirements: `pip install -r scripts/requirements.txt`

**Issue**: Out of disk space
- **Solution**: Each run needs ~1-2GB. Ensure sufficient space before starting.

**Issue**: GPU out of memory
- **Solution**: Close other GPU applications or use CPU backend (slower)

## References

See `docs/NOTES.md` Section 3.3 for theoretical background on weight decay's role in grokking.
