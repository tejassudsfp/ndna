export interface LayerSnapshot {
  layer: number;
  wo_hard: number;
  wo_soft: number;
  ff1_hard: number;
  ff1_soft: number;
}

export interface TimelineEntry {
  iter: number;
  train_loss: number;
  val_loss: number;
  lr: number;
  temperature: number;
  overall_hard_pct: number;
  overall_soft_pct: number;
  layers: LayerSnapshot[];
}

export interface KeyMoment {
  iter: number;
  title: string;
  description: string;
  type: string;
}

export interface FinalLayerStat {
  layer: number;
  wo_hard_density: number;
  wo_soft_density: number;
  ff_hard_density: number;
  ff_soft_density: number;
}

export interface Benchmarks {
  wikitext2_ppl?: number;
  wikitext103_ppl?: number;
  ptb_ppl?: number;
  lambada_acc?: number;
  lambada_ppl?: number;
  hellaswag_acc?: number;
  cbt_cn_acc?: number;
  cbt_ne_acc?: number;
  enwiki8_bpb?: number;
  text8_bpc?: number;
}

// 12 distinct layer colors — consistent across all visualizations
export const LAYER_COLORS: Record<number, string> = {
  1: "#ef4444",   // red
  2: "#f97316",   // orange
  3: "#eab308",   // yellow
  4: "#a3e635",   // lime
  5: "#22c55e",   // green
  6: "#14b8a6",   // teal
  7: "#06b6d4",   // cyan
  8: "#3b82f6",   // blue
  9: "#6366f1",   // indigo
  10: "#8b5cf6",  // violet
  11: "#d946ef",  // fuchsia
  12: "#f472b6",  // pink
};

export interface TrainingData {
  meta: {
    genome_params: number;
    total_params: number;
    masked_connections: number;
    compression_ratio: string;
    n_layers: number;
    max_iter: number;
    training_iters: number;
    checkpoint_iter: number;
  };
  final_topology: FinalLayerStat[];
  timeline: TimelineEntry[];
  key_moments: KeyMoment[];
  benchmarks: Benchmarks;
}
