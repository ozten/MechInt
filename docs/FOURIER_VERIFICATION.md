# Fourier Structure Verification

This document describes the verification infrastructure for confirming the Discrete Fourier Transform algorithm learned during grokking, as described in NOTES.md Section 5.

## Overview

The grokking phenomenon reveals that the model learns a specific mechanistic solution: the Discrete Fourier Transform (DFT) for modular arithmetic. We verify this through five key structural signatures:

1. **Embedding Clock Geometry** - Token embeddings form circles in Fourier space
2. **FFT Dominant Frequencies** - Spectral analysis shows modulus-aligned frequencies
3. **MLP Rectified Sine Waves** - Neurons encode half-wave rectified sinusoids
4. **Pairwise Manifold Rings** - Lissajous curves reveal circular structure
5. **3D Diagonal Ridges** - Constructive interference surfaces show periodic patterns

## Quick Start

### Run All Verifications

```bash
./scripts/verify_fourier_structure.sh
```

This script:
- Checks for required checkpoint files
- Validates embedding visualizations exist
- Parses FFT analysis results
- Runs comprehensive test suite (58 tests)
- Provides actionable next steps

### Expected Output

All 58 library tests should pass:
```
✓ All verification tests passed
```

## Detailed Verification Methods

### 1. Embedding Clock Verification

**Goal**: Confirm token embeddings lie on a circle in 2D Fourier space

**Implementation**: `src/analysis.rs`
- `verify_embedding_clock_geometry()` - Programmatic verification
- `project_embeddings_to_fourier_plane()` - 2D projection using DFT basis

**Metrics**:
- **Radius variance** < 10% of mean radius (tight circle)
- **Angular coverage** > 90% (no gaps)
- **Mean angle error** < 0.35 radians (ordered positioning)
- **Order fraction** > 90% (tokens in sequential order)

**Visual Check**:
```bash
# View embedding scatter plots
open artifacts/visualizations/embeddings_grokking_3x3.png
open artifacts/visualizations/embeddings_postgrok_3x3.png
```

Look for:
- Pre-grok: Random scatter (no structure)
- Post-grok: Circular patterns in 2D projections

**Tests**: `analysis::tests::embedding_clock_geometry_*`

---

### 2. FFT Dominant Frequency Verification

**Goal**: Confirm embeddings have dominant frequency components at modulus p=113

**Implementation**: `src/analysis.rs`
- `verify_fft_dominant_frequencies_from_embeddings()` - Spectral analysis
- `analyze_embeddings_fft()` - Full FFT analysis with histogram

**Metrics**:
- **Modulus frequency fraction** ≥ 15% (significant p=113 component)
- **Spectrum entropy** ≤ 0.85 (structured, not white noise)
- **Peak-to-median ratio** ≥ 2.0 (strong dominant frequencies)

**Command**:
```bash
# View FFT analysis
cat artifacts/fft_analysis_postgrok.json | jq '.dominant_frequencies[:10]'
```

**Tests**: `analysis::tests::fft_dominant_frequency_*`

---

### 3. MLP Neuron Wave Verification

**Goal**: Verify MLP neurons encode rectified sine waves (Fourier basis functions)

**Implementation**: `src/verify.rs`
- `verify_mlp_activation_waves_from_signals()` - Wave pattern detection
- `analyze_mlp_activation_wave()` - Individual neuron analysis

**Method**:
1. Extract post-ReLU activations for each neuron
2. Compute FFT to find dominant frequency
3. Generate phase-shifted rectified sine: `sin(ωt + φ)₊`
4. Compute Pearson correlation across all phases
5. Take maximum correlation as neuron wave score

**Metrics** (per neuron):
- **Negative fraction** < 0.0 (ReLU rectification ensures non-negative)
- **Dominant frequency ratio** ≥ 3.0 (clear spectral peak)
- **Rectified sine correlation** ≥ 0.7 (matches theoretical waveform)

**Tests**: `verify::tests::mlp_activation_wave_verification_*`

---

### 4. Pairwise Manifold Rings Verification

**Goal**: Confirm pairwise neuron plots form Lissajous rings (circular manifolds)

**Implementation**: `src/verify.rs`
- `verify_pairwise_manifold_transition_from_signals()` - Pre/post comparison
- `evaluate_pairwise_manifolds_from_signals()` - Circularity metrics
- `pairwise_circularity_metrics()` - Geometric analysis

**Method**:
1. For each neuron pair (i, j), plot activations as (xᵢ, xⱼ) points
2. Fit ellipse using covariance eigenvalues
3. Normalize to unit circle and compute radius variance
4. Compute circularity score = 1 / (1 + radius_CV)

**Metrics**:
- **Pre-grok average score** < 0.55 (noise, low circularity)
- **Post-grok average score** > 0.75 (clear rings)
- **Passing pairs** ≥ 10 out of 21 possible pairs (majority show rings)
- **Axis ratio** ≤ 8.0 (reasonable ellipse eccentricity)

**Transition Signature**: Sharp increase in circularity at grokking step

**Tests**: `verify::tests::pairwise_manifold_transition_*`

---

### 5. 3D Diagonal Ridge Verification

**Goal**: Verify constructive interference creates diagonal periodic surfaces

**Implementation**:
- `src/verify.rs` - `verify_constructive_interference_surface_from_grid()`
- `scripts/visualize_activation_surface.py` - Interactive 3D visualization

**Method**:
1. Extract MLP activation surface: `f(x, y)` for all input pairs
2. Compute 2D autocorrelation in 4 directions:
   - Main diagonal: `(+1, +1)`
   - Anti-diagonal: `(+1, -1)`
   - X-axis: `(+1, 0)`
   - Y-axis: `(0, +1)`
3. Compare diagonal vs axis-aligned periodicity

**Metrics**:
- **Diagonal autocorrelation** ≥ 0.2 (significant periodic structure)
- **Diagonal/axis ratio** ≥ 1.3 (diagonal dominates over axis-aligned)

**Visual Analysis**:
```bash
# Generate 3D surface plot
python scripts/visualize_activation_surface.py artifacts/activation_surface_neuron_0.json
```

Python script provides:
- Interactive Plotly 3D surface
- 2D heatmap view
- Diagonal FFT analysis
- Periodicity strength metrics

**Expected Pattern**: "Corrugated iron" - parallel diagonal ridges with period ≈ p/k for some integer k

**Tests**: `verify::tests::diagonal_ridge_verification_*`

---

## Verification Test Suite

All verification logic is validated by 58 comprehensive unit tests:

### Embedding Tests (5 tests)
- `embedding_clock_geometry_passes_on_clean_circle`
- `embedding_clock_geometry_flags_wrong_frequency`
- `fft_dominant_frequency_passes_on_modulus_structure`
- `fft_dominant_frequency_fails_on_white_noise`
- `fft_dominant_frequency_fails_on_wrong_frequency`

### MLP Wave Tests (2 tests)
- `mlp_activation_wave_verification_passes_on_rectified_sine`
- `mlp_activation_wave_verification_fails_on_noise`

### Pairwise Manifold Tests (2 tests)
- `pairwise_manifold_transition_passes_on_elliptic_signals`
- `pairwise_manifold_transition_fails_without_rings`

### Diagonal Ridge Tests (5 tests)
- `diagonal_ridge_verification_passes_on_diagonal_corrugation`
- `diagonal_ridge_verification_passes_on_anti_diagonal_pattern`
- `diagonal_ridge_verification_fails_on_axis_aligned_pattern`
- `diagonal_ridge_verification_fails_on_flat_surface`
- `diagonal_ridge_verification_fails_on_random_noise`

### Run Tests
```bash
cargo test --lib
```

---

## Integration Status

### Currently Active
- ✅ FFT analysis (integrated into main training loop)
- ✅ Embedding visualization (7x7 grids at checkpoints)
- ✅ 3D activation surfaces (exported to JSON)
- ✅ Comprehensive test coverage (58 tests)

### Available for Manual Analysis
- 🔧 Embedding clock geometry verification
- 🔧 MLP wave pattern verification
- 🔧 Pairwise manifold verification
- 🔧 Diagonal ridge verification

These functions are implemented and tested but marked `#[allow(dead_code)]` as they're reserved for post-hoc analysis or future pipeline integration.

---

## Theoretical Background

The grokking phenomenon reveals a phase transition from **memorization** to **generalization**:

### Phase 1: Memorization (steps 0-1000)
- High weight norms
- No geometric structure in embeddings
- Random scatter in pairwise plots
- Flat activation surfaces

### Phase 2: Grokking Transition (steps ~7000)
- Weight norm sharp drop
- Fourier structure emerges
- Validation accuracy jumps 1% → 90%+

### Phase 3: Generalization (steps 7000+)
- Low weight norms (stable)
- Clean circular embeddings
- Rectified sine MLP neurons
- Diagonal interference patterns

### Why Fourier Works

Modular addition has symmetry: `(a + b) mod p`

The DFT basis exploits this:
- Embedding: token `k` → `exp(2πik/p)` in frequency domain
- Addition: convolution in time = multiplication in frequency
- MLP: Fourier synthesis via weighted rectified sines

This is the **most parameter-efficient** solution, which weight decay discovers during the plateau phase.

---

## Troubleshooting

### No checkpoints found
```bash
# Run training to generate checkpoints
cargo run --release

# Or run a quick test (50 epochs)
GROK_NUM_EPOCHS=50 cargo run --release
```

### Missing visualization files
```bash
# Check if training completed
ls -lh artifacts/checkpoint_labeled/

# Re-run visualization if needed
cargo run --release
```

### FFT analysis empty
```bash
# Ensure post-grokking checkpoint exists
ls artifacts/checkpoint_labeled/postgrok_e1500/

# FFT analysis runs automatically at end of training
```

### Python visualization fails
```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Verify activation surface exported
ls artifacts/activation_surface_neuron_*.json
```

### Tests failing
```bash
# Run with verbose output
cargo test --lib -- --nocapture

# Run specific test
cargo test --lib embedding_clock
```

---

## References

- **NOTES.md Section 5**: Visual Reconstruction & Illustration Guide
- **Power et al. (2022)**: "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- **Nanda et al. (2023)**: "Progress measures for grokking via mechanistic interpretability"

---

## File Locations

```
src/
├── analysis.rs        # FFT analysis, embedding verification
├── verify.rs          # MLP waves, manifolds, diagonal ridges
├── plotting.rs        # Visualization generation
└── export.rs          # 3D surface JSON export

scripts/
├── verify_fourier_structure.sh       # Main verification script
├── visualize_activation_surface.py   # 3D surface plotter
└── README.md                          # Python setup guide

artifacts/
├── fft_analysis_postgrok.json         # Spectral analysis results
├── activation_surface_neuron_*.json   # 3D surface data
├── checkpoint_labeled/                # Model checkpoints
└── visualizations/                    # Embedding plots
```

---

## Next Steps

1. **Run full training** (requires GPU, ~2 hours):
   ```bash
   cargo run --release
   ```

2. **Examine artifacts**:
   ```bash
   ./scripts/verify_fourier_structure.sh
   ```

3. **Visual analysis**:
   ```bash
   # Embeddings
   open artifacts/visualizations/embeddings_postgrok_3x3.png

   # 3D surfaces
   python scripts/visualize_activation_surface.py artifacts/activation_surface_neuron_0.json
   ```

4. **Programmatic verification** (integrate into main.rs if needed):
   ```rust
   use crate::analysis::{verify_embedding_clock_geometry, EmbeddingClockConfig};

   let embeddings = extract_all_embeddings(&model);
   let config = EmbeddingClockConfig::default_for_modulus(113);
   let report = verify_embedding_clock_geometry(&embeddings, 113, &config)?;
   ```

---

**Status**: All verification infrastructure complete and tested. Ready for empirical validation with trained models.
