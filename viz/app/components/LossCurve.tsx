"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { TimelineEntry } from "../types";

interface Props {
  timeline: TimelineEntry[];
  currentIter: number;
}

export default function LossCurve({ timeline, currentIter }: Props) {
  const chartData = timeline.filter(
    (t) => t.iter % 500 === 0 || t.iter === currentIter
  );

  return (
    <div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
          <XAxis
            dataKey="iter"
            stroke="#333"
            tick={{ fill: "#555", fontSize: 10, fontFamily: "monospace" }}
            tickFormatter={(v) => v >= 1000 ? `${v / 1000}k` : v}
          />
          <YAxis
            stroke="#333"
            tick={{ fill: "#555", fontSize: 10, fontFamily: "monospace" }}
            domain={["auto", "auto"]}
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
              Number(value).toFixed(4),
              name === "train_loss" ? "Train" : "Val",
            ]}
          />
          <Line
            type="monotone"
            dataKey="train_loss"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            name="train_loss"
          />
          <Line
            type="monotone"
            dataKey="val_loss"
            stroke="#60a5fa"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            name="val_loss"
          />
          {currentIter > 0 && (
            <ReferenceLine
              x={currentIter}
              stroke="#3b82f6"
              strokeWidth={1}
              strokeDasharray="2 2"
              strokeOpacity={0.4}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
      <div className="flex gap-4 justify-center mt-1 font-mono text-[10px] text-[var(--muted)]">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-[1.5px] bg-[#3b82f6] inline-block" />
          Train
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-[1.5px] bg-[#60a5fa] inline-block" style={{ borderTop: "1.5px dashed #60a5fa", height: 0 }} />
          Val
        </span>
      </div>
    </div>
  );
}
