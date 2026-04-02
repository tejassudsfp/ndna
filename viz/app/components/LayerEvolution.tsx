"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { TimelineEntry, LAYER_COLORS } from "../types";

interface Props {
  timeline: TimelineEntry[];
  currentIter: number;
  nLayers: number;
  field: "wo_hard" | "ff1_hard";
  title: string;
  subtitle: string;
}

export default function LayerEvolution({
  timeline,
  currentIter,
  nLayers,
  field,
  title,
  subtitle,
}: Props) {
  // Only show data up to currentIter so lines grow with the scrubber
  const maxIter = timeline[timeline.length - 1]?.iter ?? 0;
  const chartData = timeline
    .filter(
      (t) =>
        t.iter <= currentIter &&
        (t.iter % 500 === 0 ||
          t === timeline[0] ||
          t === timeline[timeline.length - 1])
    )
    .map((entry) => {
      const point: Record<string, number> = { iter: entry.iter };
      for (const layer of entry.layers) {
        point[`L${layer.layer}`] = layer[field];
      }
      return point;
    });

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-mono text-[10px] text-[var(--muted)] uppercase tracking-wider">
            {title}
          </h3>
          <p className="font-mono text-[9px] text-[#444] mt-0.5">{subtitle}</p>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 10, bottom: 5, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
          <XAxis
            dataKey="iter"
            stroke="#333"
            tick={{ fill: "#555", fontSize: 10, fontFamily: "monospace" }}
            tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)}
            domain={[0, maxIter]}
            type="number"
          />
          <YAxis
            stroke="#333"
            tick={{ fill: "#555", fontSize: 10, fontFamily: "monospace" }}
            domain={[0, 1.05]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{
              background: "#111",
              border: "1px solid #333",
              borderRadius: 0,
              fontFamily: "monospace",
              fontSize: 11,
            }}
            labelFormatter={(v) => `Iter ${Number(v).toLocaleString()}`}
            formatter={(value, name) => [
              `${(Number(value) * 100).toFixed(1)}%`,
              String(name),
            ]}
          />
          {Array.from({ length: nLayers }, (_, i) => i + 1).map((layer) => (
            <Line
              key={layer}
              type="monotone"
              dataKey={`L${layer}`}
              stroke={LAYER_COLORS[layer]}
              strokeWidth={1.5}
              dot={false}
              name={`L${layer}`}
              opacity={0.85}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {/* Layer color legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 justify-center mt-2 font-mono text-[9px]">
        {Array.from({ length: nLayers }, (_, i) => i + 1).map((layer) => (
          <span key={layer} className="flex items-center gap-1">
            <span
              className="w-2.5 h-[2px] inline-block"
              style={{ backgroundColor: LAYER_COLORS[layer] }}
            />
            <span style={{ color: LAYER_COLORS[layer] }}>L{layer}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
