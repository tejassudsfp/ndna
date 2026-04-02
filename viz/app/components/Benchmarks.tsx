"use client";

import { motion } from "framer-motion";
import { TrainingData } from "../types";

interface Props {
  data: TrainingData;
}

interface BenchmarkDef {
  key: string;
  label: string;
  gpt2: number;
  lower_better: boolean;
  format: (v: number) => string;
  description: string;
}

const gpt2Ref: BenchmarkDef[] = [
  {
    key: "wikitext103_ppl",
    label: "WikiText-103 PPL",
    gpt2: 37.50,
    lower_better: true,
    format: (v) => v.toFixed(2),
    description:
      "Perplexity on WikiText-103, a dataset of 100M+ tokens from verified Good and Featured Wikipedia articles. Measures how well the model predicts general long-form text. Lower perplexity means the model assigns higher probability to the actual next word.",
  },
  {
    key: "ptb_ppl",
    label: "Penn Treebank PPL",
    gpt2: 65.85,
    lower_better: true,
    format: (v) => v.toFixed(2),
    description:
      "Perplexity on the Penn Treebank test set, a classic benchmark of Wall Street Journal articles. Tests language modeling on formal, edited English. Lower means better next-word prediction.",
  },
  {
    key: "lambada_ppl",
    label: "LAMBADA PPL",
    gpt2: 35.13,
    lower_better: true,
    format: (v) => v.toFixed(2),
    description:
      "Perplexity on LAMBADA, a dataset where the final word of each passage can only be guessed with broad context understanding. Tests long-range dependency modeling. Lower means the model better uses distant context to predict.",
  },
  {
    key: "lambada_acc",
    label: "LAMBADA ACC",
    gpt2: 0.4599,
    lower_better: false,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    description:
      "Accuracy on LAMBADA last-word prediction. The model must predict the exact final word. Higher means it correctly identifies the right word more often.",
  },
  {
    key: "hellaswag_acc",
    label: "HellaSwag ACC",
    gpt2: 0.312,
    lower_better: false,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    description:
      "Accuracy on HellaSwag, a commonsense reasoning benchmark. Given a scenario, the model picks the most plausible continuation from 4 choices. Tests physical and social commonsense. GPT-2 reference from Zellers et al. 2019, not from OpenAI's model card.",
  },
  {
    key: "cbt_cn_acc",
    label: "CBT Common Nouns",
    gpt2: 0.8765,
    lower_better: false,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    description:
      "Accuracy on the Children's Book Test (common nouns). The model reads a passage from a children's book and picks the correct missing common noun from 10 candidates.",
  },
  {
    key: "cbt_ne_acc",
    label: "CBT Named Entities",
    gpt2: 0.834,
    lower_better: false,
    format: (v) => `${(v * 100).toFixed(1)}%`,
    description:
      "Accuracy on the Children's Book Test (named entities). Same as CBT-CN but with character names. Tests whether the model tracks which characters do what.",
  },
  {
    key: "enwiki8_bpb",
    label: "enwiki8 BPB",
    gpt2: 1.16,
    lower_better: true,
    format: (v) => v.toFixed(3),
    description:
      "Bits per byte on enwiki8, a raw byte-level Wikipedia dump including markup. Tests compression efficiency on raw, noisy text data. Lower means better compression.",
  },
  {
    key: "text8_bpc",
    label: "text8 BPC",
    gpt2: 1.17,
    lower_better: true,
    format: (v) => v.toFixed(3),
    description:
      "Bits per character on text8, a cleaned Wikipedia dump (lowercase, spaces only). Tests character-level modeling. Lower means better character-level prediction.",
  },
];

export default function Benchmarks({ data }: Props) {
  const { benchmarks, meta } = data;

  const rows = gpt2Ref
    .filter((ref) => {
      const val = benchmarks[ref.key as keyof typeof benchmarks];
      return val != null;
    })
    .map((ref) => {
      const val = benchmarks[ref.key as keyof typeof benchmarks] as number;
      const ratio = ref.lower_better ? ref.gpt2 / val : val / ref.gpt2;
      const beats = ratio >= 1.0;
      return { ...ref, val, ratio, beats };
    });

  const wins = rows.filter((r) => r.beats);
  const beatsCount = wins.length;

  return (
    <section className="px-6 py-8">
      <div className="max-w-6xl mx-auto">
        <h2 className="font-mono text-sm text-[var(--muted)] uppercase tracking-wider mb-6">
          Benchmark Results
        </h2>

        {/* Summary */}
        <div className="border border-[var(--border)] bg-[var(--sparse)] p-5 mb-px">
          <p className="text-sm text-[var(--foreground)] leading-relaxed mb-3">
            A 354-parameter genome, trained for 52K iterations, produces a GPT-2
            that <span className="text-[#22c55e] font-bold">beats the OpenAI
            reference on {beatsCount} of {rows.length} benchmarks</span>.
            The genome controls only the wiring, not the weights. It decides
            which connections exist, then the model learns to use them.
          </p>
          <p className="text-sm text-[var(--muted)] leading-relaxed mb-4">
            The reference GPT-2 Small was trained on 40GB WebText (training
            duration not disclosed by OpenAI) with all connections always active.
            NDNA was trained for 52K iterations on OpenWebText with 33% of
            connections permanently off. Despite having fewer active connections,
            NDNA matches or exceeds GPT-2 on long-range comprehension tasks.
          </p>

          {/* Wins highlighted */}
          {wins.length > 0 && (
            <div className="border-t border-[var(--border)] pt-4">
              <h3 className="font-mono text-[10px] text-[#22c55e] uppercase tracking-wider mb-3">
                Where NDNA wins
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {wins.map((win) => (
                  <div
                    key={win.key}
                    className="border border-[#22c55e30] bg-[#22c55e08] p-3"
                  >
                    <div className="flex items-baseline justify-between mb-1">
                      <span className="font-mono text-xs font-bold text-[#22c55e]">
                        {win.label}
                      </span>
                      <span className="font-mono text-[10px] text-[#22c55e]">
                        {win.format(win.val)} vs {win.format(win.gpt2)}
                      </span>
                    </div>
                    <p className="text-[11px] text-[var(--muted)] leading-relaxed">
                      {win.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Full table */}
        <div className="border border-[var(--border)] border-t-0">
          {/* Header */}
          <div className="grid grid-cols-12 gap-px bg-[var(--border)]">
            <div className="col-span-3 bg-[var(--sparse)] px-4 py-2.5 font-mono text-[9px] text-[var(--muted)] uppercase tracking-wider">
              Benchmark
            </div>
            <div className="col-span-2 bg-[var(--sparse)] px-4 py-2.5 font-mono text-[9px] text-[var(--muted)] uppercase tracking-wider">
              Direction
            </div>
            <div className="col-span-2 bg-[var(--sparse)] px-4 py-2.5 font-mono text-[9px] text-[var(--accent)] uppercase tracking-wider">
              NDNA Genome
            </div>
            <div className="col-span-2 bg-[var(--sparse)] px-4 py-2.5 font-mono text-[9px] text-[var(--muted)] uppercase tracking-wider">
              GPT-2 Small
            </div>
            <div className="col-span-3 bg-[var(--sparse)] px-4 py-2.5 font-mono text-[9px] text-[var(--muted)] uppercase tracking-wider">
              vs Reference
            </div>
          </div>

          {/* Rows */}
          {rows.map((row, i) => (
            <motion.div
              key={row.key}
              className="grid grid-cols-12 gap-px bg-[var(--border)]"
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <div
                className={`col-span-3 px-4 py-2.5 font-mono text-xs ${
                  row.beats
                    ? "bg-[#22c55e08] text-[#22c55e] font-bold"
                    : "bg-[#0d0d0d] text-[var(--foreground)]"
                }`}
              >
                {row.label}
              </div>
              <div className="col-span-2 bg-[#0d0d0d] px-4 py-2.5 font-mono text-[10px] text-[#444]">
                {row.lower_better ? "lower better" : "higher better"}
              </div>
              <div
                className={`col-span-2 px-4 py-2.5 font-mono text-xs tabular-nums font-bold ${
                  row.beats
                    ? "bg-[#22c55e08] text-[#22c55e]"
                    : "bg-[#0d0d0d] text-[var(--accent)]"
                }`}
              >
                {row.format(row.val)}
              </div>
              <div className="col-span-2 bg-[#0d0d0d] px-4 py-2.5 font-mono text-xs text-[var(--muted)] tabular-nums">
                {row.format(row.gpt2)}
              </div>
              <div className="col-span-3 bg-[#0d0d0d] px-4 py-2.5 font-mono text-xs tabular-nums flex items-center gap-2">
                <div className="flex-1 h-2 bg-[#111] relative overflow-hidden">
                  <div
                    className="h-full transition-all duration-500"
                    style={{
                      width: `${Math.min(100, row.ratio * 100)}%`,
                      background: row.beats
                        ? "linear-gradient(90deg, #22c55e40, #22c55e)"
                        : "linear-gradient(90deg, #3b82f620, #3b82f6)",
                    }}
                  />
                </div>
                <span
                  className={
                    row.beats
                      ? "text-[#22c55e] font-bold"
                      : "text-[var(--muted)]"
                  }
                >
                  {(row.ratio * 100).toFixed(0)}%
                </span>
              </div>
            </motion.div>
          ))}
        </div>

        {/* References */}
        <div className="mt-4 space-y-2">
          <p className="font-mono text-[10px] text-[#444]">
            NDNA Genome: {meta.genome_params} parameters,{" "}
            {meta.training_iters.toLocaleString()} iterations on OpenWebText.
            Checkpoint at iter {meta.checkpoint_iter.toLocaleString()}. Hard
            masks evaluated at temperature 10.0 (fully binary).
          </p>
          <p className="font-mono text-[10px] text-[#444]">
            GPT-2 Small reference:{" "}
            <a
              href="https://huggingface.co/openai-community/gpt2"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[var(--accent)] hover:underline"
            >
              openai-community/gpt2 on HuggingFace
            </a>{" "}
            / 124M params, trained on 40GB WebText (training duration not
            disclosed). All connections always active.
          </p>
          <p className="font-mono text-[10px] text-[#444]">
            All benchmarks use sliding-window evaluation with stride 512,
            context length 1024. Perplexity (PPL) and bits-per-byte/character
            (BPB/BPC) are lower-is-better metrics measuring prediction quality.
            Accuracy (ACC) is higher-is-better.
          </p>
        </div>
      </div>
    </section>
  );
}
