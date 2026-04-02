"use client";

import { motion } from "framer-motion";
import { TrainingData } from "../types";

export default function Hero({ data }: { data: TrainingData }) {
  const { meta } = data;

  return (
    <section className="relative pt-16 pb-10 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <p className="font-mono text-[var(--muted)] text-xs mb-3 tracking-wider uppercase">
            Neural DNA / Genome-Controlled Connectivity
          </p>
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-none mb-1">
            <span className="font-mono text-[var(--accent)]">354</span>{" "}
            parameters
          </h1>
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-none mb-6">
            wire GPT-2.
          </h1>
          <p className="text-base text-[var(--muted)] max-w-2xl leading-relaxed mb-8">
            A tiny genome of {meta.genome_params} learned parameters controls
            which of {(meta.masked_connections / 1e6).toFixed(1)}M connections
            in GPT-2 are active. {meta.compression_ratio} compression.
            The genome decides the topology. The model learns to use it.
          </p>
        </motion.div>

        <motion.div
          className="grid grid-cols-2 md:grid-cols-5 gap-px border border-[var(--border)]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          {[
            { label: "Genome Params", value: meta.genome_params.toLocaleString() },
            { label: "Model Params", value: `${(meta.total_params / 1e6).toFixed(0)}M` },
            { label: "Masked Connections", value: `${(meta.masked_connections / 1e6).toFixed(1)}M` },
            { label: "Compression", value: meta.compression_ratio },
            { label: "Training Iters", value: meta.training_iters.toLocaleString() },
          ].map((stat) => (
            <div key={stat.label} className="bg-[var(--sparse)] p-4 flex flex-col">
              <span className="text-[10px] font-mono text-[var(--muted)] uppercase tracking-wider mb-1">
                {stat.label}
              </span>
              <span className="text-xl font-mono font-bold text-[var(--foreground)]">
                {stat.value}
              </span>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
