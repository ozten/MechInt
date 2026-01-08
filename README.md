# Grokking (Burn)

This project reproduces the grokking modular addition experiment in Rust using the Burn framework.

![](accuracy.png)

![](loss.png)

## Run Training (TUI)

The training loop uses Burn's `SupervisedTraining` with the default renderer. The renderer is a TUI
when the `tui` feature is enabled and stdout is a terminal; otherwise it falls back to a CLI
renderer.

1) Enable the TUI feature in `Cargo.toml`:

```toml
burn = { version = "0.20.0-pre.6", default-features = false, features = ["std", "tui", "train", "wgpu", "autodiff"] }
```

2) Run training:

```bash
cargo run --release --bin grokking
```

If you don't enable the `tui` feature, the run will still work and log metrics to the console.

## Testing

### Run All Tests

Run the complete test suite (unit tests only, no integration tests yet):

```bash
cargo test
```

Run tests in release mode (faster, recommended for large tests):

```bash
cargo test --release
```

### Run Specific Test Modules

Run only model tests:

```bash
cargo test --lib model::tests
```

Run only data tests:

```bash
cargo test --lib data::tests
```

Run only verification tests:

```bash
cargo test --lib verify::tests
```

Run analysis tests:

```bash
cargo test --lib analysis::tests
```

### Run Specific Tests

Run a single test by name:

```bash
cargo test test_forward_pass_batch_size_one
```

Run all tests matching a pattern:

```bash
cargo test batch_size
```

### Test Output Options

Show test output (useful for debugging):

```bash
cargo test -- --nocapture
```

Run tests with multiple threads (default is auto):

```bash
cargo test -- --test-threads=4
```

Run tests sequentially (single-threaded):

```bash
cargo test -- --test-threads=1
```

### Integration Tests

Currently, there are no integration tests. See BEADS issue `MechInt-1l9` for planned integration test work.

To run a mini training loop manually for testing:

```bash
# Run for just 10 epochs to verify the pipeline works
cargo run --release --bin grokking
# Then check artifacts/ for outputs
```

### Test Coverage Notes

**Known gaps** (tracked in BEADS):
- Edge case testing for `batch_size=1` (issue `MechInt-u8p`)
- Tests for `compute_restricted_loss` and `compute_excluded_loss` (issue `MechInt-efu`)
- Integration test for complete training pipeline (issue `MechInt-1l9`)
- Property-based tests for random batch sizes (issue `MechInt-5b2`)

## Outputs

- Metrics logs: `artifacts/train/` and `artifacts/valid/`
- Checkpoints: `artifacts/checkpoint/`
- Labeled checkpoints: `artifacts/checkpoint_labeled/`
- Metrics snapshots: `artifacts/loss_history.json`, `artifacts/accuracy_history.json`
- FFT analysis: `artifacts/fft_analysis.json`

## Figures and Embeddings

```bash
cargo run --release --bin generate_figures
cargo run --release --bin visualize_embeddings
```
