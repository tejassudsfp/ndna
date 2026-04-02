"use client";

import { useState } from "react";
import { TrainingData } from "../types";

interface Props {
  data: TrainingData;
}

interface PaperRef {
  authors: string;
  year: string;
  title: string;
  venue: string;
  url?: string;
}

const relatedPapers: PaperRef[] = [
  {
    authors: "Sudarshan, T.P.",
    year: "2026",
    title:
      "Neural DNA: A Compact Genome for Growing Network Architecture",
    venue: "Zenodo",
    url: "https://doi.org/10.5281/zenodo.19248389",
  },
  {
    authors: "Sudarshan, T.P.",
    year: "2026",
    title:
      "Scaling Neural DNA to GPT-2: 354 Parameters Wire a Language Model",
    venue: "Zenodo",
    url: "https://zenodo.org/records/19390927",
  },
  {
    authors: "Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.",
    year: "2019",
    title: "Language Models are Unsupervised Multitask Learners",
    venue: "OpenAI Technical Report",
    url: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
  },
  {
    authors: "Frankle, J. and Carbin, M.",
    year: "2018",
    title:
      "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks",
    venue: "ICLR",
    url: "https://arxiv.org/abs/1803.03635",
  },
  {
    authors: "Zoph, B. and Le, Q.V.",
    year: "2017",
    title: "Neural Architecture Search with Reinforcement Learning",
    venue: "ICLR",
    url: "https://arxiv.org/abs/1611.01578",
  },
  {
    authors:
      "Mocanu, D.C., Mocanu, E., Stone, P., Nguyen, P.H., Gibescu, M., Liotta, A.",
    year: "2018",
    title:
      "Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science",
    venue: "Nature Communications 9(1), 2383",
    url: "https://doi.org/10.1038/s41467-018-04316-3",
  },
  {
    authors: "Evci, U., Gale, T., Menick, J., Castro, P.S., Elsen, E.",
    year: "2020",
    title: "Rigging the Lottery: Making All Tickets Winners",
    venue: "ICML",
    url: "https://arxiv.org/abs/1911.11134",
  },
  {
    authors: "Stanley, K.O. and Miikkulainen, R.",
    year: "2002",
    title: "Evolving Neural Networks through Augmenting Topologies",
    venue: "Evolutionary Computation 10(2), 99-127",
    url: "https://doi.org/10.1162/106365602320169811",
  },
  {
    authors: "Stanley, K.O., D'Ambrosio, D.B., Gauci, J.",
    year: "2009",
    title:
      "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks",
    venue: "Artificial Life 15(2), 185-212",
    url: "https://doi.org/10.1162/artl.2009.15.2.15202",
  },
  {
    authors: "Gaier, A. and Ha, D.",
    year: "2019",
    title: "Weight Agnostic Neural Networks",
    venue: "NeurIPS",
    url: "https://arxiv.org/abs/1906.04358",
  },
  {
    authors: "Najarro, E. and Risi, S.",
    year: "2020",
    title: "Meta-Learning through Hebbian Plasticity in Random Networks",
    venue: "NeurIPS",
    url: "https://arxiv.org/abs/2007.02686",
  },
  {
    authors:
      "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.",
    year: "2017",
    title: "Attention Is All You Need",
    venue: "NeurIPS",
    url: "https://arxiv.org/abs/1706.03762",
  },
  {
    authors: "Liu, H., Simonyan, K., Yang, Y.",
    year: "2019",
    title: "DARTS: Differentiable Architecture Search",
    venue: "ICLR",
    url: "https://arxiv.org/abs/1806.09055",
  },
  {
    authors: "Han, S., Pool, J., Tung, J., Dally, W.J.",
    year: "2015",
    title:
      "Learning Both Weights and Connections for Efficient Neural Networks",
    venue: "NeurIPS",
    url: "https://arxiv.org/abs/1506.02626",
  },
  {
    authors: "LeCun, Y., Denker, J., Solla, S.",
    year: "1989",
    title: "Optimal Brain Damage",
    venue: "NeurIPS",
  },
  {
    authors:
      "Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., Peste, A.",
    year: "2021",
    title:
      "Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks",
    venue: "JMLR 22(241), 1-124",
    url: "https://jmlr.org/papers/v22/21-0366.html",
  },
  {
    authors: "Bengio, Y., Leonard, N., Courville, A.",
    year: "2013",
    title:
      "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation",
    venue: "arXiv:1308.3432",
    url: "https://arxiv.org/abs/1308.3432",
  },
  {
    authors: "Gruau, F.",
    year: "1994",
    title:
      "Neural Network Synthesis using Cellular Encoding and the Genetic Algorithm",
    venue: "PhD Thesis, Ecole Normale Superieure de Lyon",
  },
  {
    authors: "Miller, J.F.",
    year: "2004",
    title:
      "Evolving a Self-Repairing, Self-Regulating, French Flag Organism",
    venue: "GECCO",
  },
  {
    authors: "Tan, M. and Le, Q.V.",
    year: "2019",
    title:
      "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    venue: "ICML",
    url: "https://arxiv.org/abs/1905.11946",
  },
  {
    authors: "He, K., Zhang, X., Ren, S., Sun, J.",
    year: "2016",
    title: "Deep Residual Learning for Image Recognition",
    venue: "CVPR",
    url: "https://arxiv.org/abs/1512.03385",
  },
];

export default function Footer({ data }: Props) {
  const [papersOpen, setPapersOpen] = useState(false);

  return (
    <footer className="px-6 pt-10 pb-8 mt-4 border-t border-[var(--border)]">
      <div className="max-w-6xl mx-auto">
        {/* Top row: project summary + links */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* Project */}
          <div>
            <h3 className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider mb-3">
              NDNA: Neural DNA
            </h3>
            <p className="font-mono text-xs text-[#555] leading-relaxed">
              {data.meta.genome_params} parameters controlling{" "}
              {(data.meta.masked_connections / 1e6).toFixed(1)}M connections.
              Trained on OpenWebText. A100 GPU.{" "}
              {data.meta.training_iters.toLocaleString()} iterations.
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider mb-3">
              Resources
            </h3>
            <div className="space-y-1.5">
              <a
                href="https://doi.org/10.5281/zenodo.19248389"
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                Paper 1: Small-Scale (Zenodo)
              </a>
              <a
                href="https://zenodo.org/records/19390927"
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                Paper 2: Scaling to GPT-2 (Zenodo)
              </a>
              <a
                href="https://github.com/tejassudsfp/ndna"
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                GitHub Repository
              </a>
              <a
                href="https://huggingface.co/tejassuds/ndna-gpt2-small"
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                Model (HuggingFace)
              </a>
            </div>
          </div>

          {/* Author */}
          <div>
            <h3 className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider mb-3">
              Author
            </h3>
            <p className="font-mono text-xs text-[var(--foreground)] mb-1.5">
              Tejas Parthasarathi Sudarshan
            </p>
            <div className="space-y-1.5">
              <a
                href="mailto:tejas@fandesk.ai"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                tejas@fandesk.ai
              </a>
              <a
                href="https://tejassuds.com"
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                tejassuds.com
              </a>
              <a
                href="https://www.linkedin.com/in/tejassuds/"
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-xs text-[var(--accent)] hover:underline"
              >
                LinkedIn
              </a>
            </div>
          </div>
        </div>

        {/* Related Papers (expandable) */}
        <div className="border border-[var(--border)] mb-8">
          <button
            onClick={() => setPapersOpen(!papersOpen)}
            className="w-full flex items-center justify-between px-4 py-3 bg-[var(--sparse)] hover:bg-[#161616] transition-colors"
          >
            <span className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider">
              Related Papers ({relatedPapers.length})
            </span>
            <span className="font-mono text-xs text-[var(--muted)]">
              {papersOpen ? "\u2212" : "+"}
            </span>
          </button>
          {papersOpen && (
            <div className="px-4 py-3 space-y-3 max-h-[500px] overflow-y-auto">
              {relatedPapers.map((paper, i) => (
                <div key={i} className="group">
                  <p className="font-mono text-[11px] text-[var(--foreground)] leading-relaxed">
                    {paper.url ? (
                      <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[var(--accent)] hover:underline"
                      >
                        {paper.title}
                      </a>
                    ) : (
                      <span>{paper.title}</span>
                    )}
                  </p>
                  <p className="font-mono text-[10px] text-[#555]">
                    {paper.authors} ({paper.year}). {paper.venue}.
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Bottom: license + copyright */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-2 pt-4 border-t border-[var(--border)]">
          <p className="font-mono text-[10px] text-[#444]">
            &copy; {new Date().getFullYear()} Tejas Parthasarathi Sudarshan.
            Released under the{" "}
            <a
              href="https://github.com/tejassudsfp/ndna/blob/main/LICENSE"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--accent)] hover:underline"
            >
              MIT License
            </a>
            .
          </p>
          <p className="font-mono text-[10px] text-[#444]">
            Cite:{" "}
            <span className="text-[#555]">
              Sudarshan (2025). Neural DNA: A Compact Genome for Growing
              Network Architecture.
            </span>
          </p>
        </div>
      </div>
    </footer>
  );
}
