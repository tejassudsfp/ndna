#!/usr/bin/env python3
"""
Generate figures for Paper 2: Scaling Neural DNA to GPT-2.
Reads training_data.json and topology_by_temperature.json to produce publication-quality plots.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_PATH = os.path.join(SCRIPT_DIR, "viz", "public", "data", "training_data.json")
TOPO_PATH = os.path.join(SCRIPT_DIR, "results", "gpt2_full", "topology_by_temperature.json")

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_data():
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    with open(TOPO_PATH, 'r') as f:
        topo = json.load(f)
    return data, topo


def fig1_training_loss(data):
    """Validation loss over training iterations."""
    timeline = data['timeline']
    iters = [t['iter'] for t in timeline]
    val_loss = [t['val_loss'] for t in timeline]
    train_loss = [t['train_loss'] for t in timeline]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(iters, train_loss, color='#888888', linewidth=0.8, alpha=0.6, label='Train loss')
    ax.plot(iters, val_loss, color='#2a5db0', linewidth=1.5, label='Val loss')

    # Mark best checkpoint
    best_iter = 49500
    best_val = None
    for t in timeline:
        if t['iter'] == best_iter or (best_val is None and t['iter'] >= best_iter):
            best_val = t['val_loss']
            break
    if best_val is None:
        # Find closest
        closest = min(timeline, key=lambda t: abs(t['iter'] - best_iter))
        best_val = closest['val_loss']
        best_iter = closest['iter']

    ax.scatter([best_iter], [best_val], color='#c0392b', s=40, zorder=5)
    ax.annotate(f'Best: {best_val:.2f}\n(iter {best_iter:,})',
                xy=(best_iter, best_val), xytext=(best_iter - 8000, best_val + 1.2),
                fontsize=8, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Loss')
    ax.set_xlim(0, max(iters))
    ax.set_ylim(2.5, 12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_training_loss.png'), pad_inches=0.1)
    plt.close(fig)
    print("  fig1_training_loss.png")


def fig2_density_heatmap(data):
    """Per-layer hard density over training iterations as a heatmap."""
    timeline = data['timeline']
    n_layers = data['meta']['n_layers']

    # Sample every 500 iterations for clarity
    sampled = [t for t in timeline if t['iter'] % 500 == 0 or t == timeline[0] or t == timeline[-1]]

    iters = [t['iter'] for t in sampled]
    # Build matrix: rows=layers (1-12), cols=iterations
    matrix = np.zeros((n_layers, len(sampled)))
    for j, t in enumerate(sampled):
        for layer_data in t['layers']:
            i = layer_data['layer'] - 1
            # Average W_o and FF1 hard density
            matrix[i, j] = (layer_data['wo_hard'] + layer_data['ff1_hard']) / 2.0

    fig, ax = plt.subplots(figsize=(8, 4))

    # Custom colormap: black (0%) -> blue (50%) -> white (100%)
    cmap = LinearSegmentedColormap.from_list('density', [
        (0.0, '#0a0a0a'),
        (0.3, '#1a3a6e'),
        (0.5, '#2a5db0'),
        (0.7, '#6ba3e8'),
        (1.0, '#ffffff'),
    ])

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   extent=[iters[0], iters[-1], n_layers + 0.5, 0.5],
                   interpolation='nearest')

    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Layer')
    ax.set_yticks(range(1, n_layers + 1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

    cbar = fig.colorbar(im, ax=ax, label='Hard Density', shrink=0.8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Annotate key events
    ax.axvline(x=800, color='#ffffff', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.text(800 + 300, 0.7, 'Pruning\ncomplete', fontsize=7, color='white', va='top')

    ax.axvline(x=26300, color='#ffffff', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.text(26300 + 300, 0.7, 'L1\nwakes up', fontsize=7, color='white', va='top')

    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_density_heatmap.png'), pad_inches=0.1)
    plt.close(fig)
    print("  fig2_density_heatmap.png")


def fig3_final_topology(data, topo):
    """Final topology at temperature 10.0 - bar chart per layer."""
    layers_data = topo['topology_by_temperature']['10.0']

    layers = [d['layer'] for d in layers_data]
    wo_hard = [d['wo_hard_density'] * 100 for d in layers_data]
    ff_hard = [d['ff_hard_density'] * 100 for d in layers_data]

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, wo_hard, width, label='$W_O$ hard density',
                   color='#2a5db0', edgecolor='white', linewidth=0.3)
    bars2 = ax.bar(x + width/2, ff_hard, width, label='FF1 hard density',
                   color='#c0392b', edgecolor='white', linewidth=0.3, alpha=0.8)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Hard Density (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='#999999', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2)

    # Annotate the sparse layers
    for i, (wo, ff) in enumerate(zip(wo_hard, ff_hard)):
        if wo < 100:
            ax.text(i, max(wo, ff) + 2, f'{wo:.1f}%', ha='center', fontsize=7, color='#333')

    # Draw the boundary line between L4 and L5
    ax.axvline(x=3.5, color='#c0392b', linewidth=1.0, linestyle=':', alpha=0.6)
    ax.text(3.5, 105, 'sparse | dense', ha='center', fontsize=8, color='#c0392b', style='italic')

    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_final_topology.png'), pad_inches=0.1)
    plt.close(fig)
    print("  fig3_final_topology.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Loading data...")
    data, topo = load_data()
    print("Generating figures for Paper 2...")
    fig1_training_loss(data)
    fig2_density_heatmap(data)
    fig3_final_topology(data, topo)
    print("Done. All figures saved to figures/")


if __name__ == '__main__':
    main()
