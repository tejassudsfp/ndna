"use client";

import { TimelineEntry } from "../types";
import LayerEvolution from "./LayerEvolution";
import DensityHeatmap from "./DensityHeatmap";

interface Props {
  nLayers: number;
  timeline: TimelineEntry[];
  currentIter: number;
}

export default function Dashboard({
  nLayers,
  timeline,
  currentIter,
}: Props) {
  return (
    <div className="border border-[var(--border)] border-t-0">
      {/* Two line graphs side by side: W_o and FF1 hard density per layer */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-px bg-[var(--border)]">
        <div className="bg-[#0d0d0d] p-4">
          <LayerEvolution
            timeline={timeline}
            currentIter={currentIter}
            nLayers={nLayers}
            field="wo_hard"
            title="Attention Output (W_o)"
            subtitle="Hard mask density per layer over training"
          />
        </div>
        <div className="bg-[#0d0d0d] p-4">
          <LayerEvolution
            timeline={timeline}
            currentIter={currentIter}
            nLayers={nLayers}
            field="ff1_hard"
            title="Feed-Forward (FF1)"
            subtitle="Hard mask density per layer over training"
          />
        </div>
      </div>

      {/* Density heatmap full width */}
      <div className="bg-[#0d0d0d] p-4 border-t border-[var(--border)]">
        <DensityHeatmap
          timeline={timeline}
          currentIter={currentIter}
          nLayers={nLayers}
        />
      </div>
    </div>
  );
}
