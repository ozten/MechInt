#!/usr/bin/env python3
"""
3D Activation Surface Visualization

Generates interactive 3D surface plots of MLP neuron activations across all possible
input pairs (x, y) for modular addition. Reveals the 'corrugated iron' diagonal wave
structure characteristic of the Fourier algorithm discovered during grokking.

Usage:
    python scripts/visualize_activation_surface.py <json_file> [output_html]

Example:
    python scripts/visualize_activation_surface.py artifacts/activation_surface_neuron_10.json
    python scripts/visualize_activation_surface.py artifacts/activation_surface_neuron_10.json surface_10.html

Requirements:
    pip install plotly numpy
"""

import json
import sys
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def load_activation_surface(json_path):
    """Load activation surface data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_3d_surface_plot(data, output_path=None):
    """
    Create interactive 3D surface plot showing neuron activation across input grid.

    Args:
        data: Dict with keys 'neuron_index', 'modulus', and 'surface' (2D array)
        output_path: Optional path to save HTML file. If None, displays in browser.
    """
    neuron_index = data['neuron_index']
    modulus = data['modulus']
    surface = np.array(data['surface'])

    # Create coordinate grids
    x = np.arange(modulus)
    y = np.arange(modulus)

    # Create 3D surface plot
    fig = go.Figure(data=[
        go.Surface(
            x=x,
            y=y,
            z=surface,
            colorscale='Viridis',
            name=f'Neuron {neuron_index}'
        )
    ])

    fig.update_layout(
        title=f'Neuron {neuron_index} Activation Surface (Modulus {modulus})',
        scene=dict(
            xaxis_title='Input X',
            yaxis_title='Input Y',
            zaxis_title='Activation',
            xaxis=dict(range=[0, modulus-1]),
            yaxis=dict(range=[0, modulus-1]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=900,
        height=900,
        font=dict(size=12)
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved interactive plot to: {output_path}")
    else:
        fig.show()

    return fig


def create_combined_views(data, output_path=None):
    """
    Create a combined visualization with 3D surface + 2D heatmap + cross-sections.

    Args:
        data: Dict with keys 'neuron_index', 'modulus', and 'surface' (2D array)
        output_path: Optional path to save HTML file. If None, displays in browser.
    """
    from plotly.subplots import make_subplots

    neuron_index = data['neuron_index']
    modulus = data['modulus']
    surface = np.array(data['surface'])

    x = np.arange(modulus)
    y = np.arange(modulus)

    # Create subplots: 3D surface + 2D heatmap
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'3D Surface - Neuron {neuron_index}',
            f'2D Heatmap - Neuron {neuron_index}'
        ),
        specs=[[{'type': 'surface'}, {'type': 'heatmap'}]]
    )

    # Add 3D surface
    fig.add_trace(
        go.Surface(x=x, y=y, z=surface, colorscale='Viridis', showscale=True),
        row=1, col=1
    )

    # Add 2D heatmap
    fig.add_trace(
        go.Heatmap(x=x, y=y, z=surface, colorscale='Viridis', showscale=True),
        row=1, col=2
    )

    fig.update_layout(
        title=f'Activation Analysis - Neuron {neuron_index} (Modulus {modulus})',
        height=600,
        width=1400,
        showlegend=False
    )

    # Update 3D scene
    fig.update_scenes(
        xaxis_title='Input X',
        yaxis_title='Input Y',
        zaxis_title='Activation',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    )

    # Update heatmap axes
    fig.update_xaxes(title_text='Input X', row=1, col=2)
    fig.update_yaxes(title_text='Input Y', row=1, col=2)

    if output_path:
        fig.write_html(output_path)
        print(f"Saved combined visualization to: {output_path}")
    else:
        fig.show()

    return fig


def analyze_diagonal_structure(surface):
    """
    Analyze the diagonal periodicity in the activation surface.

    Returns summary statistics about diagonal wave structure.
    """
    surface = np.array(surface)
    modulus = surface.shape[0]

    # Extract main diagonal
    main_diag = np.array([surface[i, i] for i in range(modulus)])

    # Extract anti-diagonal
    anti_diag = np.array([surface[i, modulus-1-i] for i in range(modulus)])

    # Compute FFT to detect periodicity
    main_fft = np.fft.fft(main_diag)
    main_freqs = np.fft.fftfreq(len(main_diag))
    main_power = np.abs(main_fft) ** 2

    # Find dominant frequency (excluding DC component)
    dominant_idx = np.argmax(main_power[1:len(main_power)//2]) + 1
    dominant_freq = main_freqs[dominant_idx]

    stats = {
        'modulus': modulus,
        'diagonal_mean': float(np.mean(main_diag)),
        'diagonal_std': float(np.std(main_diag)),
        'diagonal_range': (float(np.min(main_diag)), float(np.max(main_diag))),
        'dominant_frequency': float(dominant_freq),
        'periodicity_strength': float(main_power[dominant_idx] / np.sum(main_power)),
    }

    return stats


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Load data
    print(f"Loading activation surface from: {json_path}")
    data = load_activation_surface(json_path)

    # Analyze structure
    stats = analyze_diagonal_structure(data['surface'])
    print(f"\nActivation Surface Statistics:")
    print(f"  Neuron Index: {data['neuron_index']}")
    print(f"  Modulus: {data['modulus']}")
    print(f"  Diagonal Mean: {stats['diagonal_mean']:.4f}")
    print(f"  Diagonal Std: {stats['diagonal_std']:.4f}")
    print(f"  Diagonal Range: [{stats['diagonal_range'][0]:.4f}, {stats['diagonal_range'][1]:.4f}]")
    print(f"  Dominant Frequency: {stats['dominant_frequency']:.4f}")
    print(f"  Periodicity Strength: {stats['periodicity_strength']:.4f}")

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        # Default: same name as input but .html extension
        output_path = json_path.with_suffix('.html')

    # Create visualization
    print(f"\nGenerating 3D surface plot...")
    create_3d_surface_plot(data, output_path=output_path)

    # Also create combined view
    combined_path = output_path.with_stem(f"{output_path.stem}_combined")
    print(f"Generating combined view...")
    create_combined_views(data, output_path=combined_path)

    print(f"\nDone! Open the HTML files in a browser to view interactive plots.")


if __name__ == '__main__':
    main()
