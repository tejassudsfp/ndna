"use client";

import { motion } from "framer-motion";
import { TrainingData } from "../types";

interface Props {
  data: TrainingData;
}

// GPT-2 small reference numbers (published baselines)
const gpt2Ref: Record<string, { value: number; label: string; lower_better: boolean }> = {
  wikitext2_ppl: { value: 29.41, label: "WikiText-2 PPL", lower_better: true },
  hellaswag_acc: { value: 0.312, label: "HellaSwag ACC", lower_better: false },
  lambada_acc: { value: 0.325, label: "LAMBADA ACC", lower_better: false },
};

export default function Stats({ data }: Props) {
  const { benchmarks, meta } = data;

  const rows: { label: string; genome: string; reference: string; pct: string }[] = [];

  if (benchmarks.wikitext2_ppl != null) {
    const ref = gpt2Ref.wikitext2_ppl;
    rows.push({
      label: ref.label,
      genome: benchmarks.wikitext2_ppl.toFixed(2),
      reference: ref.value.toFixed(2),
      pct: `${((benchmarks.wikitext2_ppl / ref.value) * 100).toFixed(0)}%`,
    });
  }

  if (benchmarks.hellaswag_acc != null) {
    const ref = gpt2Ref.hellaswag_acc;
    rows.push({
      label: ref.label,
      genome: `${(benchmarks.hellaswag_acc * 100).toFixed(1)}%`,
      reference: `${(ref.value * 100).toFixed(1)}%`,
      pct: `${((benchmarks.hellaswag_acc / ref.value) * 100).toFixed(0)}%`,
    });
  }

  if (benchmarks.lambada_acc != null) {
    const ref = gpt2Ref.lambada_acc;
    rows.push({
      label: ref.label,
      genome: `${(benchmarks.lambada_acc * 100).toFixed(1)}%`,
      reference: `${(ref.value * 100).toFixed(1)}%`,
      pct: `${((benchmarks.lambada_acc / ref.value) * 100).toFixed(0)}%`,
    });
  }

  return (
    <section className="px-6 py-8">
      <div className="max-w-5xl mx-auto">
        <h2 className="font-mono text-sm text-[var(--muted)] uppercase tracking-wider mb-4">
          Benchmark Results
        </h2>
        <div className="border border-[var(--border)]">
          {/* Header */}
          <div className="grid grid-cols-4 gap-px bg-[var(--border)]">
            {["Benchmark", "NDNA Genome", "GPT-2 Reference", "% of Ref"].map(
              (h) => (
                <div
                  key={h}
                  className="bg-[var(--sparse)] px-4 py-3 font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider"
                >
                  {h}
                </div>
              )
            )}
          </div>
          {/* Rows */}
          {rows.map((row, i) => (
            <motion.div
              key={row.label}
              className="grid grid-cols-4 gap-px bg-[var(--border)]"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="bg-[#0d0d0d] px-4 py-3 font-mono text-sm text-[var(--foreground)]">
                {row.label}
              </div>
              <div className="bg-[#0d0d0d] px-4 py-3 font-mono text-sm text-[var(--accent)] tabular-nums">
                {row.genome}
              </div>
              <div className="bg-[#0d0d0d] px-4 py-3 font-mono text-sm text-[var(--muted)] tabular-nums">
                {row.reference}
              </div>
              <div className="bg-[#0d0d0d] px-4 py-3 font-mono text-sm text-[var(--foreground)] tabular-nums">
                {row.pct}
              </div>
            </motion.div>
          ))}
        </div>
        <p className="font-mono text-[10px] text-[#444] mt-3">
          Training at {meta.training_iters.toLocaleString()} / 100,000 iterations.
          GPT-2 reference trained for ~600K+ iterations on 40GB WebText.
          Genome model uses only {meta.compression_ratio} of the parameters to control connectivity.
        </p>
      </div>
    </section>
  );
}
