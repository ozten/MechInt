#!/usr/bin/env python3
"""
Weight Decay Sweep Analysis

Analyzes the results of a weight decay sweep experiment to validate
the robustness of the grokking phenomenon.

Usage:
    python scripts/analyze_weight_decay_sweep.py <sweep_dir>

Example:
    python scripts/analyze_weight_decay_sweep.py artifacts/sweep_weight_decay_20260108_120000
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_accuracy_history(run_dir: Path) -> Optional[Dict]:
    """Load accuracy history from a run directory."""
    acc_file = run_dir / "accuracy_history.json"
    if not acc_file.exists():
        return None

    with open(acc_file, 'r') as f:
        return json.load(f)


def detect_grokking_epoch(accuracy_history: Dict) -> Optional[int]:
    """
    Detect grokking epoch from accuracy history.

    Grokking is defined as the first epoch where validation accuracy
    jumps significantly (>20%) from a plateau.
    """
    if not accuracy_history:
        return None

    val_accs = accuracy_history.get('validation_accuracies', [])
    epochs = accuracy_history.get('epochs', [])

    if len(val_accs) < 100:
        return None

    # Look for sudden jump from plateau
    MIN_PLATEAU_STEP = 100
    MIN_ACCURACY_JUMP = 20.0
    WINDOW_SIZE = 50
    TARGET_VAL_ACC = 90.0

    for i in range(MIN_PLATEAU_STEP, len(val_accs) - WINDOW_SIZE):
        # Calculate baseline (average of previous window)
        baseline = sum(val_accs[i-WINDOW_SIZE:i]) / WINDOW_SIZE

        # Check if current accuracy is significantly higher
        current = val_accs[i]

        if current > baseline + MIN_ACCURACY_JUMP and current >= TARGET_VAL_ACC:
            return epochs[i]

    return None


def analyze_run(run_dir: Path) -> Dict:
    """Analyze a single run."""
    # Extract run parameters from directory name
    run_id = run_dir.name
    parts = run_id.split('_')

    wd_str = parts[0].replace('wd', '')
    seed_str = parts[1].replace('seed', '')

    weight_decay = float(wd_str)
    seed = int(seed_str)

    # Load accuracy history
    accuracy_history = load_accuracy_history(run_dir)

    # Detect grokking
    grok_epoch = detect_grokking_epoch(accuracy_history) if accuracy_history else None

    # Get final validation accuracy
    final_val_acc = None
    if accuracy_history:
        val_accs = accuracy_history.get('validation_accuracies', [])
        if val_accs:
            final_val_acc = val_accs[-1]

    return {
        'weight_decay': weight_decay,
        'seed': seed,
        'grok_epoch': grok_epoch,
        'final_val_acc': final_val_acc,
        'success': grok_epoch is not None,
    }


def analyze_sweep(sweep_dir: Path) -> List[Dict]:
    """Analyze all runs in a sweep directory."""
    results = []

    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue

        if not run_dir.name.startswith('wd'):
            continue

        result = analyze_run(run_dir)
        results.append(result)

    return results


def print_summary(results: List[Dict]):
    """Print a summary of the sweep results."""
    print("\n" + "=" * 80)
    print("Weight Decay Sweep Analysis Summary")
    print("=" * 80)
    print()

    # Group by weight decay
    by_wd = {}
    for result in results:
        wd = result['weight_decay']
        if wd not in by_wd:
            by_wd[wd] = []
        by_wd[wd].append(result)

    # Print results for each weight decay
    for wd in sorted(by_wd.keys()):
        runs = by_wd[wd]
        print(f"Weight Decay: {wd}")
        print("-" * 40)

        success_count = sum(1 for r in runs if r['success'])
        print(f"  Success rate: {success_count}/{len(runs)}")

        grok_epochs = [r['grok_epoch'] for r in runs if r['grok_epoch'] is not None]
        if grok_epochs:
            avg_grok = sum(grok_epochs) / len(grok_epochs)
            min_grok = min(grok_epochs)
            max_grok = max(grok_epochs)
            print(f"  Grokking epoch: {avg_grok:.0f} (min: {min_grok}, max: {max_grok})")
        else:
            print(f"  Grokking epoch: N/A (no successful runs)")

        final_accs = [r['final_val_acc'] for r in runs if r['final_val_acc'] is not None]
        if final_accs:
            avg_acc = sum(final_accs) / len(final_accs)
            print(f"  Final val accuracy: {avg_acc:.1f}%")

        # Print individual runs
        print(f"\n  Individual runs:")
        for r in runs:
            status = "✅" if r['success'] else "❌"
            grok_str = f"epoch {r['grok_epoch']}" if r['grok_epoch'] else "NO GROK"
            acc_str = f"{r['final_val_acc']:.1f}%" if r['final_val_acc'] else "N/A"
            print(f"    {status} seed={r['seed']:3d}: {grok_str:15s} final_acc={acc_str}")

        print()

    print("=" * 80)


def validate_hypothesis(results: List[Dict]):
    """Validate the hypothesis about weight decay sensitivity."""
    print("\n" + "=" * 80)
    print("Hypothesis Validation")
    print("=" * 80)
    print()

    # Group by weight decay
    by_wd = {}
    for result in results:
        wd = result['weight_decay']
        if wd not in by_wd:
            by_wd[wd] = []
        by_wd[wd].append(result)

    # Expected behavior:
    # - WD=0.5: delayed or prevented grokking
    # - WD=1.0: grokking by epoch 10k
    # - WD=2.0: accelerated grokking (~epoch 5k)

    hypotheses = [
        {
            'name': 'WD=1.0 groks by epoch 10k',
            'condition': lambda: all(
                r['success'] and r['grok_epoch'] <= 10000
                for r in by_wd.get(1.0, [])
            ) if 1.0 in by_wd else False,
        },
        {
            'name': 'WD=2.0 groks earlier than WD=1.0',
            'condition': lambda: (
                all(r['success'] for r in by_wd.get(2.0, [])) and
                all(r['success'] for r in by_wd.get(1.0, [])) and
                (sum(r['grok_epoch'] for r in by_wd[2.0]) / len(by_wd[2.0])) <
                (sum(r['grok_epoch'] for r in by_wd[1.0]) / len(by_wd[1.0]))
            ) if (2.0 in by_wd and 1.0 in by_wd) else False,
        },
        {
            'name': 'WD=0.5 groks later or fails',
            'condition': lambda: (
                # Either fails to grok or groks much later
                (sum(1 for r in by_wd.get(0.5, []) if not r['success']) > 0) or
                (all(r['success'] for r in by_wd.get(0.5, [])) and
                 all(r['success'] for r in by_wd.get(1.0, [])) and
                 (sum(r['grok_epoch'] for r in by_wd[0.5]) / len(by_wd[0.5])) >
                 (sum(r['grok_epoch'] for r in by_wd[1.0]) / len(by_wd[1.0])))
            ) if (0.5 in by_wd) else False,
        },
    ]

    for h in hypotheses:
        result = "✅ PASS" if h['condition']() else "❌ FAIL"
        print(f"{result}: {h['name']}")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze weight decay sweep results')
    parser.add_argument('sweep_dir', type=str, help='Path to sweep results directory')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    print(f"Analyzing sweep: {sweep_dir}")

    results = analyze_sweep(sweep_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found {len(results)} runs")

    print_summary(results)
    validate_hypothesis(results)

    # Save results to JSON
    output_file = sweep_dir / "analysis_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
