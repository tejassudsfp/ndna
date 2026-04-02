# Neural DNA (NDNA)

A tiny learned genome that grows neural network topology. 354 parameters control 35.4 million connections in GPT-2 Small. 99,970:1 compression. Beats GPT-2 on 3 benchmarks.

**Paper 1:** [Neural DNA: A Compact Genome for Growing Network Architecture](https://doi.org/10.5281/zenodo.19248389) (Small-scale experiments)

**Paper 2:** [Scaling Neural DNA to GPT-2](https://doi.org/10.5281/zenodo.19390927) (GPT-2 experiments)

**Interactive Viz:** [ndna.tejassuds.com](https://ndna.tejassuds.com)

**Model:** [tejassuds/ndna-gpt2-small](https://huggingface.co/tejassuds/ndna-gpt2-small) on HuggingFace

## GPT-2 Results

354 genome parameters control the attention output projection (W_O) and feed-forward first layer (FF1) in all 12 transformer layers of GPT-2 Small (124M params).

![Training Loss](figures/fig1_training_loss.png)

*Validation loss over 52,700 iterations. Best checkpoint: 3.04 at iteration 49,500.*

| Benchmark | NDNA GPT-2 | GPT-2 Small | Result |
|-----------|-----------|-------------|--------|
| WikiText-103 PPL | **36.0** | 37.5 | Beats GPT-2 |
| Penn Treebank PPL | **59.4** | 65.9 | Beats GPT-2 |
| LAMBADA PPL | **22.2** | 35.1 | Beats GPT-2 |
| LAMBADA ACC | 30.8% | 46.0% | 67% of GPT-2 |
| HellaSwag ACC | 28.7% | 31.2% | 92% of GPT-2 |
| CBT-CN ACC | 82.7% | 87.7% | 94% of GPT-2 |
| CBT-NE ACC | 74.5% | 83.4% | 89% of GPT-2 |
| enwiki8 BPB | 1.39 | 1.16 | |
| text8 BPC | 1.27 | 1.17 | |

### Learned Topology

The genome discovers layer stratification: layers 5-12 are fully connected (100%), layers 1-4 are progressively pruned (98%, 45%, 20%, 8%). One-third of all masked connections are permanently off.

![Final Topology](figures/fig3_final_topology.png)

*Final per-layer hard density at temperature 10.0. Sharp boundary between sparse (L1-L4) and dense (L5-L12) zones.*

| Layer | 1 | 2 | 3 | 4 | 5-12 |
|-------|------|------|------|-----|------|
| Hard Density | 98.2% | 45.2% | 20.4% | 7.7% | 100% |

### Training Dynamics

The genome goes through four phases: over-activation (iter 0-200), aggressive pruning (200-800), stable learning (800-25K), and layer 1 resurrection (25K-50K).

![Density Heatmap](figures/fig2_density_heatmap.png)

*Per-layer density over training. Layer 1 is pruned at iter 800, then re-activates at iter 26,000.*

## Small-Scale Results (Paper 1)

| Experiment | Genome | Random Sparse | Dense | Genome vs Random |
|---|---|---|---|---|
| MNIST (MLP) | 97.54% | 97.09% | 98.33% | +0.45% |
| CIFAR-10 (MLP) | 57.14% | 51.68% | 54.32% | +5.46% |
| CIFAR-10 (CNN) | 88.93% | 85.78% | 89.80% | +3.15% |
| CIFAR-100 (Transfer) | 60.92% | 53.91% | 67.16% | +7.01% |
| IMDB (Transformer) | 85.05% | 84.66% | 84.57% | +0.39% |
| Moving MNIST (Video) | 62.23 MSE | 79.44 MSE | 62.15 MSE | -21.7% rel |

226 to 374 genome parameters control up to 2.2M connections (8,384:1 compression).

## How It Works

1. **Genome** encodes cell type embeddings (8 types, 8 dimensions) and a compatibility matrix
2. **Growth**: source and target type embeddings are compared via the compatibility matrix to produce connection probabilities
3. **Binary mask**: probabilities are thresholded to 0/1 masks (straight-through estimator for gradients)
4. **Metabolic cost**: sparsity loss forces the genome to be selective
5. **Default disconnected**: compatibility initialized negative, genome must grow every connection

The genome and network weights are trained jointly with backpropagation. Temperature annealing (1.0 to 10.0) smoothly transitions from soft to hard binary masks.

## Requirements

Python >= 3.9. GPT-2 training requires an A100 GPU. Small-scale experiments run on Apple M3 (MPS) or CPU.

```bash
pip install -r requirements.txt
```

## Quick Start

### GPT-2 with genome

```bash
# Train GPT-2 Small with NDNA genome (requires A100)
python3 experiments/rung5_gpt2.py

# Evaluate on benchmarks
python3 experiments/eval_full_benchmark.py
```

### Load pre-trained model

```python
import torch
from genome.model import Genome, GrownGPT2

# Load checkpoint
ckpt = torch.load("results/gpt2_full/genome_ckpt.pt", map_location="cpu")

# Initialize
genome = Genome(n_types=8, type_dim=8, n_bands=14)
model = GrownGPT2(genome)

# Load weights (strip torch.compile prefix)
model_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
genome_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["genome"].items()}
genome.load_state_dict(genome_state)
model.load_state_dict(model_state, strict=False)

# Enable hard binary masks for evaluation
model.hard_masks = True
model.eval()
```

### Small-scale experiments

```bash
python3 run.py train          # MNIST
python3 run.py cifar10        # CIFAR-10 MLP
python3 run.py cnn            # CIFAR-10 CNN
python3 run.py transfer100    # CIFAR-10 -> CIFAR-100
python3 run.py transformer    # IMDB sentiment
python3 run.py results        # Print all results
```

## Project Structure

```
ndna/
├── genome/
│   ├── model.py              # Genome, GrownGPT2, GrownTransformer, GrownNetwork
│   ├── baselines.py          # Dense, random sparse, dense skip baselines
│   └── visualizer.py         # Topology visualization
├── experiments/
│   ├── rung5_gpt2.py         # GPT-2 training with genome
│   ├── eval_full_benchmark.py # Full benchmark suite
│   ├── eval_gpt2.py          # HellaSwag + WikiText + topology
│   └── ...                   # Small-scale experiment scripts
├── results/
│   └── gpt2_full/            # GPT-2 eval results, logs, topology
├── viz/                      # Next.js interactive visualization
├── figures/                  # Paper figures
├── paper_ndna.md             # Paper 1: Small-scale experiments
├── paper_ndna_gpt2.md        # Paper 2: Scaling to GPT-2
└── LICENSE
```

## Citation

```bibtex
@article{sudarshan2026ndna,
  title={Neural DNA: A Compact Genome for Growing Network Architecture},
  author={Sudarshan, Tejas Parthasarathi},
  year={2026},
  doi={10.5281/zenodo.19248389}
}

@article{sudarshan2026ndna_gpt2,
  title={Scaling Neural DNA to GPT-2: 354 Parameters Wire a Language Model},
  author={Sudarshan, Tejas Parthasarathi},
  year={2026},
  doi={10.5281/zenodo.19390927}
}
```

## Author

**Tejas Parthasarathi Sudarshan**
Independent Researcher, Chennai, India
[tejas@fandesk.ai](mailto:tejas@fandesk.ai) | [tejassuds.com](https://tejassuds.com) | [LinkedIn](https://www.linkedin.com/in/tejassuds/)

## License

[MIT](LICENSE)
