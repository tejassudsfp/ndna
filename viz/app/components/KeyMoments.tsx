"use client";

import { motion } from "framer-motion";
import { KeyMoment } from "../types";

interface Props {
  moments: KeyMoment[];
  currentIter: number;
  onJump: (iter: number) => void;
}

const typeColors: Record<string, string> = {
  topology: "#3b82f6",
  density: "#f59e0b",
  temperature: "#ef4444",
  loss: "#22c55e",
};

// Extended explanations: what this means and why it matters
const inferences: Record<string, string> = {
  "Genome over-activates":
    "At initialization, the genome has random logits. Many logits happen to be positive, so the genome activates ~83% of connections. This is the genome before it has learned anything. It has no reason to prune yet.",
  "Pruning complete: 8 of 12":
    "The genome has now learned that layers 1-4 are not useful for the current training signal and turns them off entirely. Only layers 5-12 carry information forward. This is a dramatic structural decision: the genome removes a third of the network's depth and the model still keeps learning.",
  "Masks sharpening":
    "The temperature parameter controls how binary the masks are. At low temperatures, masks are soft (between 0 and 1). As temperature rises, they sharpen toward binary on/off decisions. At temperature 5.0, the genome's structural decisions are becoming increasingly committed.",
  "Layer 1 wakes up":
    "After being dead for thousands of iterations, layer 1 begins reactivating. The genome has found that with the current model weights, there is value in routing signal through layer 1 again. This shows the genome can reverse pruning decisions when the training signal changes.",
  "Topology locked":
    "At temperature 10.0, all masks are effectively binary. Every connection is either fully on or fully off. The genome can no longer make gradual adjustments. The wiring is final. From here, only the model weights continue to update.",
  "Loss breaks 4.0":
    "The model has learned enough language structure to generate coherent text. This is significant because it achieved this with ~33% of connections permanently off. The remaining connections have been sufficient for learning basic language patterns.",
  "Loss breaks 3.2":
    "The model is now approaching the quality of the fully-connected GPT-2 Small (val loss ~3.0). With only 354 genome parameters deciding which connections exist, the network has found a topology that captures most of the language modeling capacity of the full architecture.",
  "Best val loss":
    "This is the checkpoint we benchmark. The genome has converged on a topology where layers 5-12 are fully connected, layer 1 is 98% active, layer 2 is 45%, layer 3 is 20%, and layer 4 is barely alive at 8%. This sparse topology will be tested against the fully-connected GPT-2 reference.",
};

export default function KeyMoments({ moments, currentIter, onJump }: Props) {
  if (moments.length === 0) return null;

  return (
    <section className="px-6 py-8">
      <div className="max-w-6xl mx-auto">
        <h2 className="font-mono text-sm text-[var(--muted)] uppercase tracking-wider mb-2">
          Key Moments
        </h2>
        <p className="text-xs text-[var(--muted)] mb-5 max-w-2xl">
          Click any moment to jump the timeline. These mark structural
          turning points where the genome made significant wiring decisions.
        </p>
        <div className="space-y-px border border-[var(--border)]">
          {moments.map((m, i) => {
            const isPast = currentIter >= m.iter;
            const color = typeColors[m.type] ?? "#3b82f6";
            const inference =
              inferences[m.title] ??
              Object.entries(inferences).find(([k]) =>
                m.title.startsWith(k)
              )?.[1] ??
              null;

            return (
              <motion.button
                key={i}
                onClick={() => onJump(m.iter)}
                className={`w-full text-left p-4 transition-colors ${
                  isPast ? "bg-[var(--sparse)]" : "bg-[#0d0d0d]"
                } hover:bg-[#161616]`}
                whileHover={{ x: 2 }}
                whileTap={{ scale: 0.995 }}
              >
                <div className="flex items-start gap-4">
                  {/* Left: dot + iter */}
                  <div className="flex flex-col items-center flex-shrink-0 w-16 pt-0.5">
                    <span
                      className="w-2.5 h-2.5 rounded-full mb-1"
                      style={{ background: isPast ? color : "#333" }}
                    />
                    <span className="font-mono text-[10px] text-[var(--muted)]">
                      {m.iter.toLocaleString()}
                    </span>
                  </div>

                  {/* Right: content */}
                  <div className="flex-1 min-w-0">
                    <h3
                      className="font-mono text-sm font-bold mb-1"
                      style={{ color: isPast ? color : "#444" }}
                    >
                      {m.title}
                    </h3>
                    <p
                      className={`text-xs leading-relaxed mb-1 ${
                        isPast ? "text-[var(--foreground)]" : "text-[#333]"
                      }`}
                    >
                      {m.description}
                    </p>
                    {inference && (
                      <p
                        className={`text-[11px] leading-relaxed ${
                          isPast ? "text-[var(--muted)]" : "text-[#2a2a2a]"
                        }`}
                      >
                        {inference}
                      </p>
                    )}
                  </div>

                  {/* Type badge */}
                  <span
                    className="font-mono text-[8px] px-1.5 py-0.5 uppercase tracking-wider flex-shrink-0"
                    style={{
                      color: isPast ? color : "#333",
                      border: `1px solid ${isPast ? color + "40" : "#222"}`,
                    }}
                  >
                    {m.type}
                  </span>
                </div>
              </motion.button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
