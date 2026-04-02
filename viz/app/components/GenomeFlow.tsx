"use client";

import { useEffect, useRef, useCallback } from "react";
import { LayerSnapshot } from "../types";

interface Props {
  layers: LayerSnapshot[];
  nLayers: number;
  currentIter: number;
}

interface Particle {
  x: number;
  y: number;
  targetIdx: number;
  progress: number; // 0-1 between current waypoint pair
  trail: { x: number; y: number; alpha: number }[];
  speed: number;
}

export default function GenomeFlow({ layers, nLayers, currentIter }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particleRef = useRef<Particle | null>(null);
  const rafRef = useRef<number | null>(null);
  const waypointsRef = useRef<{ x: number; y: number; active: boolean; layer: number; softDensity: number }[]>([]);

  const buildWaypoints = useCallback(
    (width: number, height: number) => {
      const padding = { top: 60, bottom: 60, left: 60, right: 60 };
      const innerH = height - padding.top - padding.bottom;
      const centerX = width / 2;
      const layerSpacing = innerH / (nLayers + 1);

      // Entry point
      const points: { x: number; y: number; active: boolean; layer: number; softDensity: number }[] = [
        { x: centerX, y: padding.top - 20, active: true, layer: 0, softDensity: 1 },
      ];

      // Each layer
      for (let i = 0; i < nLayers; i++) {
        const layer = layers[i];
        const isActive = layer ? layer.wo_hard > 0 : false;
        const softDensity = layer ? (layer.wo_soft + layer.ff1_soft) / 2 : 0;
        const y = padding.top + (i + 1) * layerSpacing;

        // Active layers stay on center path; pruned layers the dot curves around
        if (isActive) {
          points.push({ x: centerX, y, active: true, layer: i + 1, softDensity });
        } else {
          // Pruned: dot veers to the side to skip
          const side = i % 2 === 0 ? -1 : 1;
          const offset = 80 + (i % 3) * 20;
          points.push({ x: centerX + side * offset, y, active: false, layer: i + 1, softDensity });
        }
      }

      // Exit point
      points.push({
        x: centerX,
        y: height - padding.bottom + 20,
        active: true,
        layer: nLayers + 1,
        softDensity: 1,
      });

      return points;
    },
    [layers, nLayers]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    waypointsRef.current = buildWaypoints(width, height);

    // Init particle at start
    if (!particleRef.current) {
      const wp = waypointsRef.current;
      particleRef.current = {
        x: wp[0].x,
        y: wp[0].y,
        targetIdx: 1,
        progress: 0,
        trail: [],
        speed: 0.012,
      };
    }

    const drawFrame = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const wp = waypointsRef.current;
      const p = particleRef.current!;

      // Clear
      ctx.fillStyle = "#0a0a0a";
      ctx.fillRect(0, 0, width, height);

      const centerX = width / 2;
      const padding = { top: 60, bottom: 60 };
      const innerH = height - padding.top - padding.bottom;
      const layerSpacing = innerH / (nLayers + 1);

      // Draw layer nodes
      for (let i = 0; i < nLayers; i++) {
        const layer = layers[i];
        if (!layer) continue;
        const isActive = layer.wo_hard > 0;
        const softDensity = (layer.wo_soft + layer.ff1_soft) / 2;
        const y = padding.top + (i + 1) * layerSpacing;

        // Connection line from previous layer
        if (i > 0) {
          ctx.strokeStyle = "#151515";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(centerX, padding.top + i * layerSpacing);
          ctx.lineTo(centerX, y);
          ctx.stroke();
        }

        if (isActive) {
          // Active layer: glowing circle
          const radius = 16 + softDensity * 10;

          // Outer glow
          const glow = ctx.createRadialGradient(centerX, y, radius * 0.5, centerX, y, radius * 2.5);
          glow.addColorStop(0, `rgba(59, 130, 246, ${0.15 * softDensity})`);
          glow.addColorStop(1, "rgba(59, 130, 246, 0)");
          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(centerX, y, radius * 2.5, 0, Math.PI * 2);
          ctx.fill();

          // Node circle
          ctx.fillStyle = `rgba(59, 130, 246, ${0.3 + softDensity * 0.5})`;
          ctx.strokeStyle = `rgba(59, 130, 246, ${0.5 + softDensity * 0.5})`;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(centerX, y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          // Label
          ctx.fillStyle = "#e5e5e5";
          ctx.font = "bold 11px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(`L${i + 1}`, centerX, y);

          // Density label to the right
          ctx.fillStyle = "#666";
          ctx.font = "10px monospace";
          ctx.textAlign = "left";
          ctx.fillText(`${(softDensity * 100).toFixed(0)}%`, centerX + radius + 12, y);
        } else {
          // Pruned layer: dim, small, with "skip" indication
          const radius = 8;

          ctx.fillStyle = "#151515";
          ctx.strokeStyle = "#222";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(centerX, y, radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          // X mark
          ctx.strokeStyle = "#333";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(centerX - 3, y - 3);
          ctx.lineTo(centerX + 3, y + 3);
          ctx.moveTo(centerX + 3, y - 3);
          ctx.lineTo(centerX - 3, y + 3);
          ctx.stroke();

          // Label
          ctx.fillStyle = "#333";
          ctx.font = "10px monospace";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(`L${i + 1}`, centerX - radius - 14, y);
        }
      }

      // Draw "INPUT" and "OUTPUT" labels
      ctx.fillStyle = "#444";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText("INPUT", centerX, padding.top - 30);
      ctx.fillText("OUTPUT", centerX, height - padding.bottom + 35);

      // Draw particle trail
      for (let i = p.trail.length - 1; i >= 0; i--) {
        const t = p.trail[i];
        t.alpha -= 0.015;
        if (t.alpha <= 0) {
          p.trail.splice(i, 1);
          continue;
        }
        ctx.fillStyle = `rgba(59, 130, 246, ${t.alpha * 0.6})`;
        ctx.beginPath();
        ctx.arc(t.x, t.y, 3 + t.alpha * 3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Update particle position
      if (wp.length > 1 && p.targetIdx < wp.length) {
        const from = wp[p.targetIdx - 1];
        const to = wp[p.targetIdx];

        p.progress += p.speed;

        if (p.progress >= 1) {
          p.progress = 0;
          p.targetIdx++;

          if (p.targetIdx >= wp.length) {
            // Loop: restart from beginning
            p.targetIdx = 1;
            p.x = wp[0].x;
            p.y = wp[0].y;
            p.trail = [];
          }
        }

        // Smooth cubic interpolation for curved path
        const t = p.progress;
        const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

        // If target is a pruned layer, curve around it
        if (!to.active && p.targetIdx < wp.length) {
          const side = to.layer % 2 === 0 ? -1 : 1;
          const curveX = centerX + side * (80 + (to.layer % 3) * 20);
          // Bezier curve: from -> control(curve) -> to
          const oneMinusT = 1 - ease;
          p.x = oneMinusT * oneMinusT * from.x + 2 * oneMinusT * ease * curveX + ease * ease * to.x;
          p.y = from.y + (to.y - from.y) * ease;
        } else {
          p.x = from.x + (to.x - from.x) * ease;
          p.y = from.y + (to.y - from.y) * ease;
        }

        // Add trail point
        p.trail.push({ x: p.x, y: p.y, alpha: 1 });

        // Cap trail length
        if (p.trail.length > 60) p.trail.shift();
      }

      // Draw particle (main dot)
      const particleGlow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 20);
      particleGlow.addColorStop(0, "rgba(255, 255, 255, 0.9)");
      particleGlow.addColorStop(0.3, "rgba(59, 130, 246, 0.6)");
      particleGlow.addColorStop(1, "rgba(59, 130, 246, 0)");
      ctx.fillStyle = particleGlow;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 20, 0, Math.PI * 2);
      ctx.fill();

      // Inner bright dot
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();

      rafRef.current = requestAnimationFrame(drawFrame);
    };

    rafRef.current = requestAnimationFrame(drawFrame);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [layers, nLayers, buildWaypoints]);

  // When layers change (timeline scrub), reset particle to re-trace the new path
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    waypointsRef.current = buildWaypoints(rect.width, rect.height);

    if (particleRef.current) {
      const wp = waypointsRef.current;
      particleRef.current.targetIdx = 1;
      particleRef.current.progress = 0;
      particleRef.current.x = wp[0]?.x ?? 0;
      particleRef.current.y = wp[0]?.y ?? 0;
      particleRef.current.trail = [];
    }
  }, [currentIter, buildWaypoints]);

  // Count active/pruned for the label
  const activeCount = layers.filter((l) => l.wo_hard > 0).length;
  const prunedCount = nLayers - activeCount;

  return (
    <section className="px-6 py-8">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-mono text-sm text-[var(--muted)] uppercase tracking-wider">
            Signal Flow
          </h2>
          <div className="font-mono text-xs text-[var(--muted)]">
            <span className="text-[var(--accent)]">{activeCount}</span> active
            {" / "}
            <span className="text-[#444]">{prunedCount}</span> pruned
          </div>
        </div>
        <div className="border border-[var(--border)] bg-[#0a0a0a]">
          <canvas
            ref={canvasRef}
            className="w-full"
            style={{ height: 700 }}
          />
        </div>
      </div>
    </section>
  );
}
