"use client";

import { TimelineEntry } from "../types";

interface Props {
  timeline: TimelineEntry[];
  currentIter: number;
  maxIter: number;
  onIterChange: (iter: number) => void;
  isPlaying: boolean;
  onPlayToggle: () => void;
}

export default function TimelineScrubber({
  timeline,
  currentIter,
  maxIter,
  onIterChange,
  isPlaying,
  onPlayToggle,
}: Props) {
  const current = timeline.find((t) => t.iter === currentIter) ?? timeline[0];

  return (
    <div className="border border-[var(--border)] bg-[var(--sparse)] p-4 mb-px">
      <div className="flex items-center gap-4">
        <button
          onClick={onPlayToggle}
          className="w-9 h-9 flex items-center justify-center border border-[var(--border)] hover:border-[var(--accent)] transition-colors flex-shrink-0"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? (
            <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor">
              <rect x="2" y="1" width="4" height="12" />
              <rect x="8" y="1" width="4" height="12" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 14 14" fill="currentColor">
              <polygon points="2,1 12,7 2,13" />
            </svg>
          )}
        </button>

        <input
          type="range"
          min={0}
          max={maxIter}
          step={100}
          value={currentIter}
          onChange={(e) => onIterChange(Number(e.target.value))}
          className="flex-1"
        />

        <div className="flex items-center gap-3 font-mono text-xs flex-shrink-0">
          <span>
            <span className="text-[var(--accent)] font-bold">
              {currentIter.toLocaleString()}
            </span>
            <span className="text-[var(--muted)]">
              /{maxIter.toLocaleString()}
            </span>
          </span>
          <span className="text-[var(--muted)]">|</span>
          <span className="text-[var(--muted)]">
            T={current?.temperature?.toFixed(1) ?? "0"}
          </span>
          <span className="text-[var(--muted)]">|</span>
          <span className="text-[var(--muted)]">
            {current?.overall_hard_pct?.toFixed(1) ?? "0"}% hard
          </span>
        </div>
      </div>
    </div>
  );
}
