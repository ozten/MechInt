#!/usr/bin/env bash
#
# Weight Decay Sweep for Grokking Robustness Validation
# =======================================================
#
# This script validates the robustness of the grokking phenomenon across
# different weight decay values and random seeds.
#
# Experiment Design (from NOTES.md Section 3.3):
# - Baseline: weight_decay = 1.0 (current spec)
# - Lower: weight_decay = 0.5 (should delay or prevent grokking)
# - Higher: weight_decay = 2.0 (should accelerate grokking)
# - Control: different random seeds (42, 123, 456)
#
# Success Criteria:
# - All runs with WD=1.0 should grok by epoch 10k
# - WD=0.5 may fail to grok by epoch 20k
# - WD=2.0 should grok earlier (~epoch 5k)
#
# Estimated Compute: 9 runs × 10-15 hours = 90-135 GPU hours
#
# Usage:
#   ./scripts/sweep_weight_decay.sh [--dry-run] [--quick]
#
# Options:
#   --dry-run: Print commands without executing
#   --quick: Run short experiments (100 epochs) for testing
#

set -euo pipefail

# Configuration
WEIGHT_DECAYS=(0.5 1.0 2.0)
SEEDS=(42 123 456)
DEFAULT_EPOCHS=10000
QUICK_EPOCHS=100

# Parse arguments
DRY_RUN=false
QUICK=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--dry-run] [--quick]"
            exit 1
            ;;
    esac
done

if [ "$QUICK" = true ]; then
    EPOCHS=$QUICK_EPOCHS
    echo "🚀 Quick mode: running with $EPOCHS epochs"
else
    EPOCHS=$DEFAULT_EPOCHS
    echo "🚀 Full mode: running with $EPOCHS epochs"
fi

# Create output directory
SWEEP_DIR="artifacts/sweep_weight_decay_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

echo "📊 Weight Decay Sweep Experiment"
echo "================================"
echo ""
echo "Configuration:"
echo "  - Weight decays: ${WEIGHT_DECAYS[*]}"
echo "  - Seeds: ${SEEDS[*]}"
echo "  - Epochs: $EPOCHS"
echo "  - Total runs: $((${#WEIGHT_DECAYS[@]} * ${#SEEDS[@]}))"
echo "  - Output directory: $SWEEP_DIR"
echo ""

# Create run log
RUN_LOG="$SWEEP_DIR/run_log.txt"
echo "Weight Decay Sweep - $(date)" > "$RUN_LOG"
echo "================================" >> "$RUN_LOG"
echo "" >> "$RUN_LOG"

# Run counter
run_num=0
total_runs=$((${#WEIGHT_DECAYS[@]} * ${#SEEDS[@]}))

# Main experiment loop
for wd in "${WEIGHT_DECAYS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_num=$((run_num + 1))

        # Create run-specific subdirectory
        run_id="wd${wd}_seed${seed}"
        run_dir="$SWEEP_DIR/$run_id"
        mkdir -p "$run_dir"

        echo ""
        echo "[$run_num/$total_runs] Running: weight_decay=$wd, seed=$seed"
        echo "  Output: $run_dir"

        # Log this run
        echo "[$run_num/$total_runs] weight_decay=$wd, seed=$seed -> $run_dir" >> "$RUN_LOG"

        # Prepare command
        cmd="GROK_WEIGHT_DECAY=$wd GROK_SEED=$seed GROK_NUM_EPOCHS=$EPOCHS GROK_SKIP_VALIDATION=1 cargo run --release --bin grokking"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] $cmd"
            echo "  [DRY RUN] Would move artifacts to: $run_dir"
        else
            # Execute training
            echo "  Starting training at $(date +%H:%M:%S)..."
            start_time=$(date +%s)

            # Run with output capture
            if eval "$cmd" > "$run_dir/training_output.log" 2>&1; then
                end_time=$(date +%s)
                duration=$((end_time - start_time))
                hours=$((duration / 3600))
                minutes=$(((duration % 3600) / 60))

                echo "  ✅ Completed in ${hours}h ${minutes}m"
                echo "  ✅ Completed in ${hours}h ${minutes}m" >> "$RUN_LOG"

                # Move artifacts to run-specific directory
                if [ -d "artifacts" ]; then
                    # Copy all generated artifacts
                    cp -r artifacts/*.json "$run_dir/" 2>/dev/null || true
                    cp -r artifacts/checkpoint_* "$run_dir/" 2>/dev/null || true
                    cp -r artifacts/visualizations "$run_dir/" 2>/dev/null || true
                fi

                # Extract grokking step from output if available
                if grep -q "Grokking detected at epoch" "$run_dir/training_output.log"; then
                    grok_info=$(grep "Grokking detected at epoch" "$run_dir/training_output.log" | head -1)
                    echo "  📈 $grok_info"
                    echo "  📈 $grok_info" >> "$RUN_LOG"
                else
                    echo "  ⚠️  No grokking detected"
                    echo "  ⚠️  No grokking detected" >> "$RUN_LOG"
                fi
            else
                echo "  ❌ Training failed (check $run_dir/training_output.log)"
                echo "  ❌ Training failed" >> "$RUN_LOG"
            fi
        fi
    done
done

echo ""
echo "================================"
echo "✅ Sweep completed!"
echo ""
echo "Results saved to: $SWEEP_DIR"
echo "Run log: $RUN_LOG"
echo ""

if [ "$DRY_RUN" = false ]; then
    echo "Next steps:"
    echo "  1. Analyze results: python scripts/analyze_weight_decay_sweep.py $SWEEP_DIR"
    echo "  2. Generate comparison plots: python scripts/plot_sweep_comparison.py $SWEEP_DIR"
    echo ""
fi
