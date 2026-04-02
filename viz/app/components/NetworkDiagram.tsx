"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { LayerSnapshot } from "../types";

interface Props {
  layers: LayerSnapshot[];
  nLayers: number;
}

export default function NetworkDiagram({ layers, nLayers }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || layers.length === 0) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const height = 320;

    svg.attr("viewBox", `0 0 ${width} ${height}`);
    svg.selectAll("*").remove();

    const margin = { top: 24, right: 30, bottom: 10, left: 56 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const layerHeight = innerH / nLayers;
    const blockWidth = innerW / 2 - 12;

    const colorScale = (density: number) => {
      if (density < 0.01) return "#111";
      if (density < 0.5) return d3.interpolateRgb("#111", "#3b82f6")(density * 2);
      return d3.interpolateRgb("#3b82f6", "#e5e5e5")((density - 0.5) * 2);
    };

    // Column headers
    g.append("text")
      .attr("x", blockWidth / 2)
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .attr("fill", "#555")
      .attr("font-size", "9px")
      .attr("font-family", "var(--font-geist-mono), monospace")
      .text("W_o (Attention)");

    g.append("text")
      .attr("x", blockWidth + 24 + blockWidth / 2)
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .attr("fill", "#555")
      .attr("font-size", "9px")
      .attr("font-family", "var(--font-geist-mono), monospace")
      .text("FF1 (Feed-Forward)");

    layers.forEach((layer, i) => {
      const y = i * layerHeight;
      const h = layerHeight - 3;

      // Layer label
      g.append("text")
        .attr("x", -8)
        .attr("y", y + h / 2 + 3)
        .attr("text-anchor", "end")
        .attr("fill", layer.wo_hard > 0 ? "#999" : "#333")
        .attr("font-size", "10px")
        .attr("font-family", "var(--font-geist-mono), monospace")
        .text(`L${layer.layer}`);

      // W_o block
      g.append("rect")
        .attr("x", 0)
        .attr("y", y)
        .attr("width", blockWidth)
        .attr("height", h)
        .attr("fill", colorScale(layer.wo_soft))
        .attr("stroke", layer.wo_hard > 0 ? "#3b82f640" : "#1a1a1a")
        .attr("stroke-width", layer.wo_hard > 0 ? 1 : 0.5);

      // FF1 block
      g.append("rect")
        .attr("x", blockWidth + 24)
        .attr("y", y)
        .attr("width", blockWidth)
        .attr("height", h)
        .attr("fill", colorScale(layer.ff1_soft))
        .attr("stroke", layer.ff1_hard > 0 ? "#3b82f640" : "#1a1a1a")
        .attr("stroke-width", layer.ff1_hard > 0 ? 1 : 0.5);

      // Density text
      const woText = layer.wo_hard > 0 ? `${(layer.wo_soft * 100).toFixed(0)}%` : "OFF";
      const ffText = layer.ff1_hard > 0 ? `${(layer.ff1_soft * 100).toFixed(0)}%` : "OFF";

      g.append("text")
        .attr("x", blockWidth / 2)
        .attr("y", y + h / 2 + 3)
        .attr("text-anchor", "middle")
        .attr("fill", layer.wo_hard > 0 ? (layer.wo_soft > 0.6 ? "#0a0a0a" : "#ccc") : "#2a2a2a")
        .attr("font-size", "10px")
        .attr("font-family", "var(--font-geist-mono), monospace")
        .attr("font-weight", "bold")
        .text(woText);

      g.append("text")
        .attr("x", blockWidth + 24 + blockWidth / 2)
        .attr("y", y + h / 2 + 3)
        .attr("text-anchor", "middle")
        .attr("fill", layer.ff1_hard > 0 ? (layer.ff1_soft > 0.6 ? "#0a0a0a" : "#ccc") : "#2a2a2a")
        .attr("font-size", "10px")
        .attr("font-family", "var(--font-geist-mono), monospace")
        .attr("font-weight", "bold")
        .text(ffText);
    });
  }, [layers, nLayers]);

  return (
    <svg ref={svgRef} className="w-full" style={{ height: 320 }} />
  );
}
