"use client";

import { motion } from "framer-motion";

export default function HowItWorks() {
  return (
    <section className="px-6 py-10">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
        >
          <h2 className="font-mono text-sm text-[var(--muted)] uppercase tracking-wider mb-6">
            How NDNA Works
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-px border border-[var(--border)]">
            {/* GPT-2 side */}
            <div className="bg-[var(--sparse)] p-5">
              <h3 className="font-mono text-xs font-bold text-[var(--muted)] mb-3">
                Standard GPT-2 Small
              </h3>
              <p className="text-sm text-[var(--muted)] leading-relaxed mb-3">
                Every connection is always active. All 12 transformer layers are fully
                connected at all times. 124M parameters, all trained independently.
                The architecture is fixed before training begins.
              </p>
              <div className="font-mono text-[10px] text-[#444] space-y-1">
                <p>124M trainable parameters</p>
                <p>All connections always active</p>
                <p>Fixed architecture, no learned topology</p>
                <p>Trained on 40GB WebText (duration undisclosed)</p>
              </div>
            </div>

            {/* NDNA side */}
            <div className="bg-[#0d0d0d] p-5 border-l border-[var(--border)]">
              <h3 className="font-mono text-xs font-bold text-[var(--accent)] mb-3">
                NDNA Genome GPT-2
              </h3>
              <p className="text-sm text-[var(--foreground)] leading-relaxed mb-3">
                A 354-parameter genome controls which of 35.4M connections
                are active. The genome learns the wiring. The model learns to use
                whatever wiring the genome provides. The architecture emerges from
                training.
              </p>
              <div className="font-mono text-[10px] text-[var(--muted)] space-y-1">
                <p>354 genome parameters control 35.4M connections</p>
                <p>Genome decides which connections exist</p>
                <p>Topology emerges and evolves during training</p>
                <p>~52K iterations on OpenWebText</p>
              </div>
            </div>
          </div>

          {/* Mask types explanation */}
          <div className="mt-px border border-[var(--border)] border-t-0">
            <div className="bg-[var(--sparse)] px-5 py-3">
              <h3 className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider mb-3">
                Connection States
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Hard ON */}
                <div className="flex gap-3">
                  <div className="w-3 h-3 mt-0.5 flex-shrink-0 bg-[var(--foreground)]" />
                  <div>
                    <p className="font-mono text-xs font-bold text-[var(--foreground)] mb-1">
                      Hard mask ON
                    </p>
                    <p className="text-[11px] text-[var(--muted)] leading-relaxed">
                      Connection is permanently active. Signal passes through
                      fully. The genome has decided this connection matters. This
                      is a binary decision: the underlying logit is positive.
                    </p>
                  </div>
                </div>

                {/* Soft / partial */}
                <div className="flex gap-3">
                  <div className="w-3 h-3 mt-0.5 flex-shrink-0 bg-[var(--accent)]" />
                  <div>
                    <p className="font-mono text-xs font-bold text-[var(--accent)] mb-1">
                      Soft mask (partial)
                    </p>
                    <p className="text-[11px] text-[var(--muted)] leading-relaxed">
                      Connection is partially active. Signal is scaled by a value
                      between 0 and 1 (sigmoid of genome logit times temperature).
                      As temperature rises during training, soft masks get pushed
                      toward fully on or fully off.
                    </p>
                  </div>
                </div>

                {/* OFF */}
                <div className="flex gap-3">
                  <div className="w-3 h-3 mt-0.5 flex-shrink-0 bg-[#222] border border-[#333]" />
                  <div>
                    <p className="font-mono text-xs font-bold text-[#555] mb-1">
                      Hard mask OFF
                    </p>
                    <p className="text-[11px] text-[var(--muted)] leading-relaxed">
                      Connection is dead. No signal passes through. The genome
                      has decided this connection does not contribute to
                      performance. The underlying logit is negative. The weight
                      matrix still exists but its output is zeroed.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
