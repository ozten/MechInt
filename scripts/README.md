# 3D Activation Surface Visualization

This directory contains tools for visualizing MLP neuron activation patterns that emerge during grokking.

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
