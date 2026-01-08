#!/usr/bin/env python3
"""
Weight Decay Sweep Comparison Plots

Generates comparison plots for weight decay sweep experiments.

Usage:
    python scripts/plot_sweep_comparison.py <sweep_dir>

Example:
    python scripts/plot_sweep_comparison.py artifacts/sweep_weight_decay_20260108_120000
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly not installed. Run: pip install plotly")
    sys.exit(1)


def load_run_data(run_dir: Path) -> Dict:
    """Load accuracy and loss history from a run directory."""
    # Extract run parameters
    run_id = run_dir.name
    parts = run_id.split('_')

    wd_str = parts[0].replace('wd', '')
    seed_str = parts[1].replace('seed', '')

    weight_decay = float(wd_str)
    seed = int(seed_str)

    # Load accuracy history
    acc_file = run_dir / "accuracy_history.json"
    accuracy_history = None
    if acc_file.exists():
        with open(acc_file, 'r') as f:
            accuracy_history = json.load(f)

    # Load loss history
    loss_file = run_dir / "loss_history.json"
    loss_history = None
    if loss_file.exists():
        with open(loss_file, 'r') as f:
            loss_history = json.load(f)

    return {
        'weight_decay': weight_decay,
        'seed': seed,
        'accuracy': accuracy_history,
        'loss': loss_history,
    }


def plot_validation_accuracy_comparison(runs: List[Dict], output_file: Path):
    """Plot validation accuracy curves grouped by weight decay."""
    fig = go.Figure()

    # Group by weight decay
    by_wd = {}
    for run in runs:
        wd = run['weight_decay']
        if wd not in by_wd:
            by_wd[wd] = []
        by_wd[wd].append(run)

    # Color scheme
    colors = {
        0.5: 'rgb(255, 127, 14)',  # Orange
        1.0: 'rgb(44, 160, 44)',    # Green
        2.0: 'rgb(31, 119, 180)',   # Blue
    }

    # Plot each run
    for wd in sorted(by_wd.keys()):
        runs_for_wd = by_wd[wd]

        for run in runs_for_wd:
            if not run['accuracy']:
                continue

            epochs = run['accuracy'].get('epochs', [])
            val_accs = run['accuracy'].get('validation_accuracies', [])

            if not epochs or not val_accs:
                continue

            # Add trace
            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_accs,
                mode='lines',
                name=f'WD={wd}, seed={run["seed"]}',
                line=dict(color=colors.get(wd, 'gray'), width=2),
                opacity=0.7,
                legendgroup=f'wd{wd}',
            ))

    fig.update_layout(
        title='Validation Accuracy Comparison Across Weight Decay Values',
        xaxis_title='Epoch',
        yaxis_title='Validation Accuracy (%)',
        xaxis_type='log',
        yaxis_type='linear',
        hovermode='closest',
        width=1200,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
    )

    # Add horizontal line at 90% (grokking threshold)
    fig.add_hline(y=90, line_dash="dash", line_color="gray",
                  annotation_text="Grokking threshold (90%)")

    fig.write_html(str(output_file))
    print(f"Saved: {output_file}")


def plot_grokking_epoch_comparison(runs: List[Dict], output_file: Path):
    """Plot grokking epochs as a box plot grouped by weight decay."""
    # Group by weight decay and extract grokking epochs
    by_wd = {}
    for run in runs:
        wd = run['weight_decay']
        if wd not in by_wd:
            by_wd[wd] = {'epochs': [], 'seeds': []}

        # Detect grokking epoch
        if run['accuracy']:
            val_accs = run['accuracy'].get('validation_accuracies', [])
            epochs = run['accuracy'].get('epochs', [])

            # Simple detection: first epoch where val_acc >= 90%
            for i, acc in enumerate(val_accs):
                if acc >= 90:
                    by_wd[wd]['epochs'].append(epochs[i])
                    by_wd[wd]['seeds'].append(run['seed'])
                    break

    # Create box plot
    fig = go.Figure()

    for wd in sorted(by_wd.keys()):
        epochs = by_wd[wd]['epochs']

        if not epochs:
            continue

        fig.add_trace(go.Box(
            y=epochs,
            name=f'WD={wd}',
            boxmean='sd',  # Show mean and std dev
        ))

    fig.update_layout(
        title='Grokking Epoch Distribution by Weight Decay',
        xaxis_title='Weight Decay',
        yaxis_title='Grokking Epoch',
        yaxis_type='linear',
        width=800,
        height=600,
    )

    fig.write_html(str(output_file))
    print(f"Saved: {output_file}")


def plot_combined_comparison(runs: List[Dict], output_file: Path):
    """Create a combined figure with multiple comparison plots."""
    # Group by weight decay
    by_wd = {}
    for run in runs:
        wd = run['weight_decay']
        if wd not in by_wd:
            by_wd[wd] = []
        by_wd[wd].append(run)

    # Create subplots: 2 rows x 2 cols
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Validation Accuracy (log scale)',
            'Training Loss (log scale)',
            'Validation Loss (log scale)',
            'Grokking Epoch Distribution',
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "box"}],
        ],
    )

    # Color scheme
    colors = {
        0.5: 'rgb(255, 127, 14)',  # Orange
        1.0: 'rgb(44, 160, 44)',    # Green
        2.0: 'rgb(31, 119, 180)',   # Blue
    }

    # Plot validation accuracy (row 1, col 1)
    for wd in sorted(by_wd.keys()):
        for run in by_wd[wd]:
            if not run['accuracy']:
                continue

            epochs = run['accuracy'].get('epochs', [])
            val_accs = run['accuracy'].get('validation_accuracies', [])

            if not epochs or not val_accs:
                continue

            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_accs,
                mode='lines',
                name=f'WD={wd}',
                line=dict(color=colors.get(wd, 'gray'), width=1.5),
                opacity=0.6,
                legendgroup=f'wd{wd}',
                showlegend=True,
            ), row=1, col=1)

    # Plot training loss (row 1, col 2)
    for wd in sorted(by_wd.keys()):
        for run in by_wd[wd]:
            if not run['loss']:
                continue

            steps = run['loss'].get('steps', [])
            train_losses = run['loss'].get('training_losses', [])

            if not steps or not train_losses:
                continue

            fig.add_trace(go.Scatter(
                x=steps,
                y=train_losses,
                mode='lines',
                name=f'WD={wd}',
                line=dict(color=colors.get(wd, 'gray'), width=1.5),
                opacity=0.6,
                legendgroup=f'wd{wd}',
                showlegend=False,
            ), row=1, col=2)

    # Plot validation loss (row 2, col 1)
    for wd in sorted(by_wd.keys()):
        for run in by_wd[wd]:
            if not run['loss']:
                continue

            steps = run['loss'].get('steps', [])
            val_losses = run['loss'].get('validation_losses', [])

            if not steps or not val_losses:
                continue

            fig.add_trace(go.Scatter(
                x=steps,
                y=val_losses,
                mode='lines',
                name=f'WD={wd}',
                line=dict(color=colors.get(wd, 'gray'), width=1.5),
                opacity=0.6,
                legendgroup=f'wd{wd}',
                showlegend=False,
            ), row=2, col=1)

    # Plot grokking epoch distribution (row 2, col 2)
    grok_data = {}
    for wd in sorted(by_wd.keys()):
        grok_epochs = []

        for run in by_wd[wd]:
            if not run['accuracy']:
                continue

            val_accs = run['accuracy'].get('validation_accuracies', [])
            epochs = run['accuracy'].get('epochs', [])

            for i, acc in enumerate(val_accs):
                if acc >= 90:
                    grok_epochs.append(epochs[i])
                    break

        if grok_epochs:
            fig.add_trace(go.Box(
                y=grok_epochs,
                name=f'WD={wd}',
                marker_color=colors.get(wd, 'gray'),
                legendgroup=f'wd{wd}',
                showlegend=False,
            ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Epoch", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Step", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Step", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Weight Decay", row=2, col=2)

    fig.update_yaxes(title_text="Val Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Train Loss", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Val Loss", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Grokking Epoch", row=2, col=2)

    fig.update_layout(
        title_text='Weight Decay Sweep: Comprehensive Comparison',
        height=1000,
        width=1400,
        showlegend=True,
    )

    fig.write_html(str(output_file))
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot weight decay sweep comparisons')
    parser.add_argument('sweep_dir', type=str, help='Path to sweep results directory')
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)

    if not sweep_dir.exists():
        print(f"Error: Directory not found: {sweep_dir}")
        sys.exit(1)

    print(f"Plotting sweep: {sweep_dir}")

    # Load all run data
    runs = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith('.'):
            continue

        if not run_dir.name.startswith('wd'):
            continue

        run_data = load_run_data(run_dir)
        runs.append(run_data)

    if not runs:
        print("No runs found!")
        sys.exit(1)

    print(f"Found {len(runs)} runs")

    # Create output directory for plots
    plot_dir = sweep_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    plot_validation_accuracy_comparison(
        runs,
        plot_dir / "validation_accuracy_comparison.html"
    )

    plot_grokking_epoch_comparison(
        runs,
        plot_dir / "grokking_epoch_distribution.html"
    )

    plot_combined_comparison(
        runs,
        plot_dir / "combined_comparison.html"
    )

    print(f"\n✅ All plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
