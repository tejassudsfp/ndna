"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TrainingData } from "./types";
import Hero from "./components/Hero";
import HowItWorks from "./components/HowItWorks";
import TimelineScrubber from "./components/TimelineScrubber";
import Dashboard from "./components/Dashboard";
import KeyMoments from "./components/KeyMoments";
import Benchmarks from "./components/Benchmarks";
import Footer from "./components/Footer";

export default function Page() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [currentIter, setCurrentIter] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    fetch("/data/training_data.json")
      .then((r) => r.json())
      .then((d: TrainingData) => {
        setData(d);
        setCurrentIter(d.meta.max_iter);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!isPlaying || !data) return;
    const maxIter = data.meta.max_iter;
    const step = 100;
    const msPerStep = 40;

    const animate = (timestamp: number) => {
      if (timestamp - lastTimeRef.current >= msPerStep) {
        lastTimeRef.current = timestamp;
        setCurrentIter((prev) => {
          const next = prev + step;
          if (next > maxIter) {
            setIsPlaying(false);
            return maxIter;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying, data]);

  const handlePlayToggle = useCallback(() => {
    if (!data) return;
    setIsPlaying((prev) => {
      if (!prev) {
        setCurrentIter((iter) => (iter >= data.meta.max_iter ? 0 : iter));
        lastTimeRef.current = 0;
      }
      return !prev;
    });
  }, [data]);

  const handleIterChange = useCallback((iter: number) => {
    setIsPlaying(false);
    setCurrentIter(iter);
  }, []);

  const handleJumpToMoment = useCallback((iter: number) => {
    setIsPlaying(false);
    setCurrentIter(iter);
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="font-mono text-red-500">Failed to load data: {error}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="font-mono text-[var(--muted)] text-sm animate-pulse">
          Loading genome data...
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen pb-16">
      <Hero data={data} />

      <HowItWorks />

      {/* Evolution Dashboard: scrubber + per-layer line graphs + heatmap */}
      <section className="px-6 py-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="font-mono text-sm text-[var(--muted)] uppercase tracking-wider mb-4">
            Genome Evolution
          </h2>
          <TimelineScrubber
            timeline={data.timeline}
            currentIter={currentIter}
            maxIter={data.meta.max_iter}
            onIterChange={handleIterChange}
            isPlaying={isPlaying}
            onPlayToggle={handlePlayToggle}
          />
          <Dashboard
            nLayers={data.meta.n_layers}
            timeline={data.timeline}
            currentIter={currentIter}
          />
        </div>
      </section>

      <KeyMoments
        moments={data.key_moments}
        currentIter={currentIter}
        onJump={handleJumpToMoment}
      />

      <Benchmarks data={data} />

      <Footer data={data} />
    </main>
  );
}
