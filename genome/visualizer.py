"""
Genome visualization dashboard.

Generates a multi-panel figure showing the genome's growth rules,
cell types, emergent topology, and network diagram.
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math


def get_band_labels(dims):
    labels = []
    for i, d in enumerate(dims):
        if i == 0:
            labels.append(f"input\n({d})")
        elif i == len(dims) - 1:
            labels.append(f"output\n({d})")
        else:
            labels.append(f"band {i}\n({d})")
    return labels


def show_dashboard(genome, model, save_path="results/genome_viz.png"):
    """Generate a full visual dashboard of the genome."""
    genome.eval()
    dims = model.dims
    n_bands = len(dims)
    labels = get_band_labels(dims)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("GENOME DASHBOARD", fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35,
                           top=0.93, bottom=0.05, left=0.05, right=0.95)

    # 1. COMPATIBILITY MATRIX
    ax1 = fig.add_subplot(gs[0, 0])
    with torch.no_grad():
        rule = genome.connection_rule().cpu().numpy()
    im = ax1.imshow(rule, cmap='RdBu_r', aspect='equal')
    ax1.set_title("Type Compatibility\n(connection rule)", fontsize=11)
    ax1.set_xlabel("Source type")
    ax1.set_ylabel("Target type")
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # 2. AFFINITY SIMILARITY
    ax2 = fig.add_subplot(gs[0, 1])
    with torch.no_grad():
        aff = genome.affinity.cpu()
        sim = (aff @ aff.T / math.sqrt(genome.type_dim)).numpy()
    im = ax2.imshow(sim, cmap='viridis', aspect='equal')
    ax2.set_title("Type Affinity\n(similarity)", fontsize=11)
    ax2.set_xlabel("Type")
    ax2.set_ylabel("Type")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # 3. TOPOLOGY MAP
    ax3 = fig.add_subplot(gs[0, 2])
    topo = np.zeros((n_bands, n_bands))
    with torch.no_grad():
        for t in range(1, n_bands):
            for s in range(t):
                m = genome.growth_mask(s, t, dims[s], dims[t])
                topo[t, s] = m.mean().item()
    im = ax3.imshow(topo, cmap='hot', aspect='equal', vmin=0, vmax=0.5)
    ax3.set_title("Topology Map\n(soft density per band pair)", fontsize=11)
    ax3.set_xticks(range(n_bands))
    ax3.set_yticks(range(n_bands))
    ax3.set_xticklabels([l.split('\n')[0] for l in labels], fontsize=8, rotation=45)
    ax3.set_yticklabels([l.split('\n')[0] for l in labels], fontsize=8)
    ax3.set_xlabel("Source band")
    ax3.set_ylabel("Target band")
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # 4. GENOME STATS
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    _, total, sd = model.count_effective()
    gp = sum(p.numel() for p in genome.parameters())
    stats_text = (
        f"Genome params: {gp}\n"
        f"Possible connections: {total:,}\n"
        f"Compression: {total // gp}:1\n"
        f"Soft density: {sd:.1%}\n"
        f"Depth penalty: {F.softplus(genome.depth_penalty).item():.2f}\n"
        f"Connection scale: {F.softplus(genome.connection_scale).item():.2f}\n"
        f"\nCell types: {genome.n_types}\n"
        f"Type dim: {genome.type_dim}\n"
        f"Bands: {n_bands}\n"
        f"Dims: {dims}"
    )
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title("Genome Stats", fontsize=11)

    # 5-8. GROWTH MASKS (top 4 by density)
    mask_pairs = []
    with torch.no_grad():
        for t in range(1, n_bands):
            for s in range(t):
                m = genome.growth_mask(s, t, dims[s], dims[t])
                avg = m.mean().item()
                if avg > 0.01:
                    mask_pairs.append((s, t, m.cpu().numpy(), avg))

    mask_pairs.sort(key=lambda x: x[3], reverse=True)
    for idx, (s, t, mask, avg) in enumerate(mask_pairs[:4]):
        ax = fig.add_subplot(gs[1, idx])
        if mask.shape[0] > 48 or mask.shape[1] > 48:
            step_r = max(1, mask.shape[0] // 48)
            step_c = max(1, mask.shape[1] // 48)
            mask_show = mask[::step_r, ::step_c]
        else:
            mask_show = mask
        im = ax.imshow(mask_show, cmap='inferno', aspect='auto', vmin=0, vmax=0.6)
        src_l = labels[s].split('\n')[0]
        tgt_l = labels[t].split('\n')[0]
        ax.set_title(f"{src_l} -> {tgt_l}\navg={avg:.3f}", fontsize=10)
        ax.set_xlabel(f"src neurons ({dims[s]})")
        ax.set_ylabel(f"tgt neurons ({dims[t]})")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for idx in range(len(mask_pairs[:4]), 4):
        ax = fig.add_subplot(gs[1, idx])
        ax.axis('off')
        ax.text(0.5, 0.5, "No connectivity", ha='center', va='center',
                fontsize=11, color='gray')

    # 9. CELL TYPE DISTRIBUTIONS
    ax9 = fig.add_subplot(gs[2, 0:2])
    with torch.no_grad():
        for band_idx in range(n_bands):
            n = min(dims[band_idx], 48)
            td = genome.type_distribution(band_idx, n).cpu().numpy()
            dominant = td.argmax(axis=1)
            positions = np.linspace(0, 1, n)
            ax9.scatter(positions, [band_idx] * n, c=dominant, cmap='Set1',
                        vmin=0, vmax=genome.n_types - 1, s=20, alpha=0.8)

    ax9.set_yticks(range(n_bands))
    ax9.set_yticklabels([l.split('\n')[0] for l in labels], fontsize=9)
    ax9.set_xlabel("Position within band (0=left, 1=right)")
    ax9.set_title("Cell Type Map\n(dominant type per neuron position)", fontsize=11)
    ax9.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap='Set1',
                                norm=plt.Normalize(vmin=0, vmax=genome.n_types - 1))
    cb = plt.colorbar(sm, ax=ax9, fraction=0.02)
    cb.set_label("Cell type")

    # 10. TYPE DISTRIBUTION HEATMAP
    ax10 = fig.add_subplot(gs[2, 2])
    with torch.no_grad():
        td = genome.type_distribution(1, dims[1]).cpu().numpy()
    im = ax10.imshow(td.T, cmap='YlOrRd', aspect='auto')
    ax10.set_title(f"Band 1 Type Distribution\n({dims[1]} neurons x {genome.n_types} types)", fontsize=10)
    ax10.set_xlabel("Neuron position")
    ax10.set_ylabel("Cell type")
    plt.colorbar(im, ax=ax10, fraction=0.046)

    # 11. NETWORK DIAGRAM
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.axis('off')

    band_y = np.linspace(0.9, 0.1, n_bands)
    band_x_center = 0.5
    max_width = 0.8

    for i, (y, d) in enumerate(zip(band_y, dims)):
        width = max_width * min(d, 100) / 784
        rect = plt.Rectangle((band_x_center - width / 2, y - 0.015),
                              width, 0.03, facecolor='steelblue', alpha=0.7)
        ax11.add_patch(rect)
        label = labels[i].replace('\n', ' ')
        ax11.text(band_x_center, y, label, ha='center', va='center',
                  fontsize=8, fontweight='bold', color='white')

    with torch.no_grad():
        for t in range(1, n_bands):
            for s in range(t):
                m = genome.growth_mask(s, t, dims[s], dims[t])
                avg = m.mean().item()
                if avg > 0.01:
                    lw = avg * 8
                    alpha = min(avg * 3, 0.8)
                    color = 'red' if (t - s) > 1 else 'gray'
                    ax11.annotate("", xy=(band_x_center + 0.05 * (s - t), band_y[t] + 0.02),
                                  xytext=(band_x_center + 0.05 * (s - t), band_y[s] - 0.02),
                                  arrowprops=dict(arrowstyle='->', lw=lw,
                                                  color=color, alpha=alpha))

    ax11.set_xlim(0, 1)
    ax11.set_ylim(0, 1)
    ax11.set_title("Network Diagram\n(red = skip connections)", fontsize=11)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")
    plt.close(fig)
