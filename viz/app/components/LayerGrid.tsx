"use client";

import { motion } from "framer-motion";
import { LayerSnapshot } from "../types";

interface Props {
  layers: LayerSnapshot[];
}

function DensityBar({
  label,
  density,
  isActive,
}: {
  label: string;
  density: number;
  isActive: boolean;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-[9px] text-[var(--muted)] w-5">{label}</span>
      <div className="flex-1 h-2.5 bg-[#111] relative overflow-hidden">
        <motion.div
          className="h-full"
          style={{
            background: isActive
              ? `linear-gradient(90deg, #1e3a5f, #3b82f6)`
              : "#1a1a1a",
          }}
          initial={{ width: 0 }}
          animate={{ width: `${density * 100}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
        />
      </div>
      <span className="font-mono text-[9px] text-[var(--muted)] w-7 text-right tabular-nums">
        {isActive ? `${(density * 100).toFixed(0)}%` : "OFF"}
      </span>
    </div>
  );
}

export default function LayerGrid({ layers }: Props) {
  return (
    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-px bg-[var(--border)]">
      {layers.map((layer) => {
        const isActive = layer.wo_hard > 0;
        return (
          <div
            key={layer.layer}
            className={`p-3 ${isActive ? "bg-[#0d0d0d]" : "bg-[#090909]"}`}
          >
            <div className="flex items-center justify-between mb-2">
              <span
                className={`font-mono text-xs font-bold ${
                  isActive ? "text-[var(--foreground)]" : "text-[#2a2a2a]"
                }`}
              >
                L{layer.layer}
              </span>
              {isActive ? (
                <span className="font-mono text-[8px] px-1 py-px bg-[#3b82f615] text-[var(--accent)] border border-[#3b82f630]">
                  ON
                </span>
              ) : (
                <span className="font-mono text-[8px] px-1 py-px bg-[#111] text-[#333] border border-[#1a1a1a]">
                  OFF
                </span>
              )}
            </div>
            <div className="space-y-1">
              <DensityBar label="Wo" density={layer.wo_soft} isActive={isActive} />
              <DensityBar label="FF" density={layer.ff1_soft} isActive={isActive} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
