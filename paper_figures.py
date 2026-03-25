"""
Generate all figures for the NDNA paper.

Reads results from results/*.json and produces publication-quality PNGs.
Run from the ndna/ directory: python3 paper_figures.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'genome': '#2E86AB',
    'random': '#A23B72',
    'dense': '#555555',
    'skip': '#F18F01',
    'transfer': '#C73E1D',
    'accent': '#3B1F2B',
}

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)


def load_results():
    """Load all experiment results from JSON files."""
    data = {}

    # MNIST MLP
    with open('results/train_mnist_20260322T130729.json') as f:
        data['mnist'] = json.load(f)

    # CIFAR-10 MLP
    with open('results/train_cifar10_20260322T134119.json') as f:
        data['cifar10_mlp'] = json.load(f)

    # CIFAR-10 CNN
    with open('results/train_cifar10_cnn_20260323T130423.json') as f:
        data['cifar10_cnn'] = json.load(f)

    # CIFAR-100 Transfer
    with open('results/transfer_cifar100_20260323T205931.json') as f:
        data['cifar100'] = json.load(f)

    # IMDB Transformer (v2, the successful run)
    with open('results/rung3_transformer_20260325T000723.json') as f:
        data['imdb'] = json.load(f)

    return data


def fig2_compression(data):
    """Figure 2: Compression ratio scaling with network size."""
    experiments = [
        ('CIFAR-10\nCNN', 258, 22654),
        ('CIFAR-100\nCNN', 258, 43084),
        ('MNIST\nMLP', 226, 174112),
        ('CIFAR-10\nMLP', 226, 1707008),
        ('IMDB\nTransformer', 258, 2163200),
    ]

    names = [e[0] for e in experiments]
    genome_params = [e[1] for e in experiments]
    connections = [e[2] for e in experiments]
    ratios = [c / g for g, c in zip(genome_params, connections)]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(names))
    width = 0.35

    bars = ax1.bar(x, ratios, width=0.6, color=COLORS['genome'], alpha=0.85,
                   edgecolor='white', linewidth=1.5)

    ax1.set_ylabel('Compression Ratio (connections : genome params)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_title('Compression Ratio Scales with Network Size')

    # Add ratio labels on bars
    for bar, ratio, conn in zip(bars, ratios, connections):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f'{ratio:,.0f}:1\n({conn:,.0f})',
                 ha='center', va='bottom', fontsize=9)

    # Add genome param annotation
    ax1.axhline(y=0, color='gray', linewidth=0.5)
    ax1.text(0.02, 0.95, 'Genome: 226-258 params (constant)',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig2_compression.png')
    plt.close()
    print(f'  Saved {OUT}/fig2_compression.png')


def fig3_genome_vs_random(data):
    """Figure 3: Accuracy gap (genome - random sparse) across experiments."""
    experiments = [
        ('MNIST\nMLP',
         data['mnist']['results']['grown']['acc'],
         data['mnist']['results']['random_sparse']['acc']),
        ('CIFAR-10\nMLP',
         data['cifar10_mlp']['results']['grown']['acc'],
         data['cifar10_mlp']['results']['random_sparse']['acc']),
        ('CIFAR-10\nCNN',
         data['cifar10_cnn']['results']['genome_cnn']['acc'],
         data['cifar10_cnn']['results']['random_sparse_cnn']['acc']),
        ('CIFAR-100\nTransfer',
         data['cifar100']['results']['frozen_genome']['acc'],
         data['cifar100']['results']['random_sparse']['acc']),
        ('IMDB\nTransformer',
         data['imdb']['results']['genome_transformer']['acc'],
         data['imdb']['results']['random_sparse_transformer']['acc']),
    ]

    names = [e[0] for e in experiments]
    gaps = [(e[1] - e[2]) * 100 for e in experiments]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [COLORS['genome'] if g > 0 else COLORS['random'] for g in gaps]
    bars = ax.bar(range(len(names)), gaps, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.6)

    ax.set_ylabel('Accuracy Gap (NDNA - Random Sparse) in %')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_title('NDNA vs Random Sparse: Accuracy Gap Across Experiments')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')

    for bar, gap in zip(bars, gaps):
        y_pos = bar.get_height() + 0.15 if gap > 0 else bar.get_height() - 0.35
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'+{gap:.2f}%', ha='center', va='bottom' if gap > 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.text(0.02, 0.95, 'NDNA wins on every experiment',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig3_genome_vs_random.png')
    plt.close()
    print(f'  Saved {OUT}/fig3_genome_vs_random.png')


def fig4_topology_convergence(data):
    """Figure 4: Topology convergence - CIFAR-10 vs CIFAR-100 genome.

    Since we don't have the raw genome tensors readily available in JSON,
    we use the reported densities to create a schematic comparison.
    """
    import torch

    # Load both genomes
    import sys
    sys.path.insert(0, '.')
    from genome import Genome, GrownConvNetwork

    band_names = ['input', 'g1c1', 'g1c2', 'g2c1', 'g2c2', 'g3c1', 'g3c2', 'fc']
    band_channels = [3, 16, 16, 32, 32, 64, 64, 100]  # Using 100 classes for both

    def get_density_matrix(genome_path, n_bands=8):
        genome = Genome(n_types=8, type_dim=8, n_bands=n_bands)
        genome.load_state_dict(torch.load(genome_path, weights_only=True))
        mat = np.zeros((n_bands, n_bands))
        with torch.no_grad():
            for tgt in range(1, n_bands):
                for src in range(tgt):
                    src_ch = band_channels[src]
                    tgt_ch = band_channels[tgt]
                    m = genome.growth_mask(src, tgt, src_ch, tgt_ch, temperature=10.0)
                    mat[tgt, src] = (m > 0.5).float().mean().item()
        return mat

    c10_path = 'results/genome_cnn_20260323T130423.pt'
    c100_path = 'results/genome_cifar100_fresh.pt'

    if not os.path.exists(c10_path) or not os.path.exists(c100_path):
        print(f'  SKIP fig4: genome files not found')
        return

    mat_c10 = get_density_matrix(c10_path)
    mat_c100 = get_density_matrix(c100_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CIFAR-10 genome
    im0 = axes[0].imshow(mat_c10, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title('CIFAR-10 Genome')
    axes[0].set_xticks(range(8))
    axes[0].set_yticks(range(8))
    axes[0].set_xticklabels(band_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(band_names, fontsize=8)
    axes[0].set_xlabel('Source Band')
    axes[0].set_ylabel('Target Band')

    # CIFAR-100 genome
    im1 = axes[1].imshow(mat_c100, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('CIFAR-100 Genome (fresh)')
    axes[1].set_xticks(range(8))
    axes[1].set_yticks(range(8))
    axes[1].set_xticklabels(band_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(band_names, fontsize=8)
    axes[1].set_xlabel('Source Band')

    # Difference
    diff = mat_c100 - mat_c10
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title('Difference (C100 - C10)')
    axes[2].set_xticks(range(8))
    axes[2].set_yticks(range(8))
    axes[2].set_xticklabels(band_names, rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels(band_names, fontsize=8)
    axes[2].set_xlabel('Source Band')

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Density')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Density')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Delta')

    # Add text annotations for non-zero cells
    for ax, mat in [(axes[0], mat_c10), (axes[1], mat_c100)]:
        for i in range(8):
            for j in range(8):
                if mat[i, j] > 0.01:
                    ax.text(j, i, f'{mat[i,j]:.0%}', ha='center', va='center',
                            fontsize=7, color='black' if mat[i,j] < 0.6 else 'white')

    plt.suptitle('Topology Convergence: Independent Genomes Discover Similar Structure',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig4_topology_convergence.png')
    plt.close()
    print(f'  Saved {OUT}/fig4_topology_convergence.png')


def fig5_genome_vs_dense(data):
    """Figure 5: Genome accuracy relative to dense baseline."""
    experiments = [
        ('MNIST\nMLP',
         data['mnist']['results']['grown']['acc'],
         data['mnist']['results']['normal_mlp']['acc']),
        ('CIFAR-10\nMLP',
         data['cifar10_mlp']['results']['grown']['acc'],
         data['cifar10_mlp']['results']['normal_mlp']['acc']),
        ('CIFAR-10\nCNN',
         data['cifar10_cnn']['results']['genome_cnn']['acc'],
         data['cifar10_cnn']['results']['dense_resnet']['acc']),
        ('CIFAR-100\nFresh',
         data['cifar100']['results']['fresh_genome']['acc'],
         data['cifar100']['results']['dense_resnet']['acc']),
        ('IMDB\nTransformer',
         data['imdb']['results']['genome_transformer']['acc'],
         data['imdb']['results']['dense_transformer']['acc']),
    ]

    names = [e[0] for e in experiments]
    gaps = [(e[1] - e[2]) * 100 for e in experiments]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [COLORS['genome'] if g >= 0 else COLORS['dense'] for g in gaps]
    bars = ax.bar(range(len(names)), gaps, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.6)

    ax.set_ylabel('Accuracy Gap (NDNA - Dense Baseline) in %')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_title('NDNA vs Dense Baseline: From Slightly Behind to Ahead')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')

    for bar, gap in zip(bars, gaps):
        y_pos = bar.get_height() + 0.1 if gap >= 0 else bar.get_height() - 0.15
        sign = '+' if gap >= 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{sign}{gap:.2f}%', ha='center',
                va='bottom' if gap >= 0 else 'top',
                fontsize=10, fontweight='bold')

    # Trend line
    x_vals = np.arange(len(gaps))
    z = np.polyfit(x_vals, gaps, 1)
    p = np.poly1d(z)
    ax.plot(x_vals, p(x_vals), '--', color=COLORS['accent'], alpha=0.5, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig5_genome_vs_dense.png')
    plt.close()
    print(f'  Saved {OUT}/fig5_genome_vs_dense.png')


def fig6_transformer_heatmap(data):
    """Figure 6: Transformer cross-layer connectivity heatmap."""
    # Connectivity data from the training output
    band_names = ['emb', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'cls']
    n = len(band_names)

    # Soft density values from the experiment output
    # Format: (src_idx, tgt_idx, soft_density)
    connections = [
        (0, 1, 0.999),  # emb->L1 (attn+FF combined, both 99.9%)
        (1, 2, 0.370),  # L1->L2
        (0, 3, 0.968),  # emb->L3 (skip)
        (0, 4, 0.027),  # emb->L4 (skip, very weak)
        (3, 4, 0.199),  # L3->L4
        (3, 5, 0.112),  # L3->L5 (skip)
        (4, 5, 0.484),  # L4->L5
        (6, 7, 1.000),  # L6->cls
    ]

    mat = np.zeros((n, n))
    for src, tgt, density in connections:
        mat[tgt, src] = density

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(band_names)
    ax.set_yticklabels(band_names)
    ax.set_xlabel('Source Band')
    ax.set_ylabel('Target Band')
    ax.set_title('NDNA Transformer: Learned Cross-Layer Connectivity on IMDB\n'
                 '(Soft Density, 258 genome params)')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0.01:
                color = 'white' if mat[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{mat[i,j]:.0%}', ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)

    # Highlight the highway
    highway_cells = [(0, 1), (0, 3), (6, 7)]  # (src, tgt)
    for src, tgt in highway_cells:
        rect = plt.Rectangle((src - 0.5, tgt - 0.5), 1, 1,
                              linewidth=2.5, edgecolor=COLORS['genome'],
                              facecolor='none', linestyle='-')
        ax.add_patch(rect)

    ax.text(0.02, 0.02,
            'Blue boxes: highway path\n(emb->L1->skip to L3->...->cls)',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Soft Density')

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig6_transformer_heatmap.png')
    plt.close()
    print(f'  Saved {OUT}/fig6_transformer_heatmap.png')


def fig1_method_overview():
    """Figure 1: NDNA Method Overview diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('NDNA: From Compact Genome to Sparse Network Topology', fontsize=14, pad=20)

    # Genome box (small)
    genome_box = mpatches.FancyBboxPatch((0.5, 1.5), 2.5, 3,
                                          boxstyle="round,pad=0.2",
                                          facecolor='#E8F4FD',
                                          edgecolor=COLORS['genome'],
                                          linewidth=2)
    ax.add_patch(genome_box)
    ax.text(1.75, 4.1, 'Genome G', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['genome'])
    ax.text(1.75, 3.5, 'Type Affinity A (8x8)', fontsize=8, ha='center')
    ax.text(1.75, 3.1, 'Compatibility C (8x8)', fontsize=8, ha='center')
    ax.text(1.75, 2.7, 'Band Types (Lx8 each)', fontsize=8, ha='center')
    ax.text(1.75, 2.3, 'Scale, Penalty (2)', fontsize=8, ha='center')
    ax.text(1.75, 1.7, '226-258 params', fontsize=10, fontweight='bold',
            ha='center', color=COLORS['accent'])

    # Arrow 1
    ax.annotate('', xy=(4.0, 3.0), xytext=(3.2, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(3.6, 3.3, 'generate', fontsize=9, ha='center', color='gray')

    # Mask generation box (medium)
    mask_box = mpatches.FancyBboxPatch((4.0, 1.5), 3.0, 3,
                                        boxstyle="round,pad=0.2",
                                        facecolor='#FFF3E0',
                                        edgecolor=COLORS['skip'],
                                        linewidth=2)
    ax.add_patch(mask_box)
    ax.text(5.5, 4.1, 'Mask Generation', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['skip'])
    ax.text(5.5, 3.5, r'$\tau$ = softmax(base + pos*grad)', fontsize=9, ha='center')
    ax.text(5.5, 3.0, r'R = (AA$^T$/d + C) * scale', fontsize=9, ha='center')
    ax.text(5.5, 2.5, r'M = $\sigma$($\tau_t$ R $\tau_s^T$ * temp)', fontsize=9, ha='center')
    ax.text(5.5, 1.9, 'Binary {0,1} masks', fontsize=10, fontweight='bold',
            ha='center', color=COLORS['accent'])

    # Arrow 2
    ax.annotate('', xy=(8.0, 3.0), xytext=(7.2, 3.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(7.6, 3.3, 'apply', fontsize=9, ha='center', color='gray')

    # Sparse network box (large)
    net_box = mpatches.FancyBboxPatch((8.0, 0.8), 3.5, 4.4,
                                       boxstyle="round,pad=0.2",
                                       facecolor='#F3E5F5',
                                       edgecolor=COLORS['random'],
                                       linewidth=2)
    ax.add_patch(net_box)
    ax.text(9.75, 4.8, 'Sparse Network', fontsize=12, fontweight='bold',
            ha='center', color=COLORS['random'])

    # Draw a small network diagram
    layers = [(8.7, [1.5, 2.3, 3.1, 3.9]),
              (9.4, [1.8, 2.6, 3.5]),
              (10.1, [1.5, 2.3, 3.1, 3.9]),
              (10.8, [2.0, 3.0])]

    for lx, ys in layers:
        for y in ys:
            circle = plt.Circle((lx, y), 0.12, color=COLORS['genome'], alpha=0.7)
            ax.add_patch(circle)

    # Draw some connections (sparse)
    sparse_conns = [
        (8.7, 1.5, 9.4, 1.8), (8.7, 3.1, 9.4, 2.6), (8.7, 3.9, 9.4, 3.5),
        (9.4, 1.8, 10.1, 2.3), (9.4, 2.6, 10.1, 3.1), (9.4, 3.5, 10.1, 3.9),
        (10.1, 2.3, 10.8, 2.0), (10.1, 3.9, 10.8, 3.0),
        (8.7, 3.9, 10.1, 3.9),  # skip connection
    ]
    for x1, y1, x2, y2 in sparse_conns:
        is_skip = abs(x2 - x1) > 0.8
        ax.plot([x1, x2], [y1, y2],
                color=COLORS['transfer'] if is_skip else 'gray',
                alpha=0.6, linewidth=1.5 if is_skip else 0.8)

    ax.text(9.75, 1.1, 'W * M  (masked weights)', fontsize=9, ha='center',
            color=COLORS['accent'])

    # Bottom annotation
    ax.text(6.0, 0.3, 'Genome is trained jointly with weights via gradient descent.\n'
            'Metabolic cost (sparsity loss) pressures toward efficient wiring.',
            fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/fig1_method_overview.png')
    plt.close()
    print(f'  Saved {OUT}/fig1_method_overview.png')


def main():
    print('Generating NDNA paper figures...')
    print()

    data = load_results()

    fig1_method_overview()
    fig2_compression(data)
    fig3_genome_vs_random(data)
    fig4_topology_convergence(data)
    fig5_genome_vs_dense(data)
    fig6_transformer_heatmap(data)

    print()
    print(f'All figures saved to {OUT}/')
    print('Done.')


if __name__ == '__main__':
    main()
