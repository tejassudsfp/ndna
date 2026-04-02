"use client";

import { useEffect, useRef, useState } from "react";
import { TimelineEntry, LAYER_COLORS } from "../types";

interface Props {
  timeline: TimelineEntry[];
  currentIter: number;
  nLayers: number;
}

// Parse hex color to RGB
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.substring(0, 2), 16),
    parseInt(h.substring(2, 4), 16),
    parseInt(h.substring(4, 6), 16),
  ];
}

export default function DensityHeatmap({
  timeline,
  currentIter,
  nLayers,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showHard, setShowHard] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const sampled = timeline.filter(
      (t) =>
        t.iter % 500 === 0 ||
        t === timeline[0] ||
        t === timeline[timeline.length - 1]
    );

    const cols = sampled.length;
    const rows = nLayers;
    const cellW = Math.floor(canvas.width / cols);
    const cellH = Math.floor(canvas.height / rows);

    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    sampled.forEach((entry, col) => {
      const isCurrentCol = Math.abs(entry.iter - currentIter) < 300;

      entry.layers.forEach((layer, row) => {
        const density = showHard
          ? (layer.wo_hard + layer.ff1_hard) / 2
          : (layer.wo_soft + layer.ff1_soft) / 2;

        // Use layer color, modulated by density
        const [lr, lg, lb] = hexToRgb(LAYER_COLORS[layer.layer] ?? "#3b82f6");

        let r: number, g: number, b: number;
        if (density < 0.01) {
          r = 13;
          g = 13;
          b = 13;
        } else {
          // Interpolate from dark (#111) to the layer's color at full density
          const t = Math.min(1, density);
          r = Math.round(13 + (lr - 13) * t);
          g = Math.round(13 + (lg - 13) * t);
          b = Math.round(13 + (lb - 13) * t);
        }

        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(col * cellW, row * cellH, cellW - 1, cellH - 1);
      });

      if (isCurrentCol) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.strokeRect(col * cellW, 0, cellW, canvas.height);
      }
    });
  }, [timeline, currentIter, nLayers, showHard]);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider">
            Density Heatmap
          </h3>
          <p className="font-mono text-[9px] text-[#444] mt-0.5">
            Combined W_o + FF1 density per layer over training. Each row is a layer, colored by its identity.
          </p>
        </div>
        <div className="flex gap-1.5 font-mono text-[10px]">
          <button
            className={`px-2 py-0.5 border transition-colors ${
              !showHard
                ? "border-[var(--accent)] text-[var(--accent)]"
                : "border-[var(--border)] text-[var(--muted)]"
            }`}
            onClick={() => setShowHard(false)}
          >
            Soft
          </button>
          <button
            className={`px-2 py-0.5 border transition-colors ${
              showHard
                ? "border-[var(--accent)] text-[var(--accent)]"
                : "border-[var(--border)] text-[var(--muted)]"
            }`}
            onClick={() => setShowHard(true)}
          >
            Hard
          </button>
        </div>
      </div>
      <div className="flex gap-3">
        <div className="flex flex-col justify-between py-0.5 font-mono text-[9px]">
          {Array.from({ length: nLayers }, (_, i) => (
            <span
              key={i}
              className="leading-none"
              style={{ color: LAYER_COLORS[i + 1] }}
            >
              L{i + 1}
            </span>
          ))}
        </div>
        <canvas
          ref={canvasRef}
          width={800}
          height={192}
          className="w-full"
          style={{ imageRendering: "pixelated" }}
        />
      </div>
      <div className="flex justify-between mt-1.5 font-mono text-[9px] text-[var(--muted)] pl-7">
        <span>Iter 0</span>
        <span>
          {(timeline[timeline.length - 1]?.iter / 1000).toFixed(0)}K
        </span>
      </div>
    </div>
  );
}
