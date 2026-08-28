import { describe, expect, it } from "vitest";
import { buildSeries, seriesColor } from "./series";
import type { ScoreHistoryEntry } from "../types";

const entry = (node: string, step: number, score: number, at: string): ScoreHistoryEntry => ({
  node_name: node,
  step,
  risk_score: score,
  predicted_at: at,
});

describe("buildSeries", () => {
  it("groups history entries by node name", () => {
    const series = buildSeries([
      entry("agent", 0, 0.1, "2026-01-01T00:00:00Z"),
      entry("store", 0, 0.2, "2026-01-01T00:00:00Z"),
      entry("agent", 1, 0.3, "2026-01-01T00:01:00Z"),
    ]);
    expect(series.map((s) => s.name).sort()).toEqual(["agent", "store"]);
    expect(series.find((s) => s.name === "agent")?.points).toHaveLength(2);
  });

  it("sorts each series' points chronologically regardless of input order", () => {
    const series = buildSeries([
      entry("agent", 1, 0.3, "2026-01-01T00:01:00Z"),
      entry("agent", 0, 0.1, "2026-01-01T00:00:00Z"),
    ]);
    const points = series[0].points;
    expect(points[0].y).toBe(0.1);
    expect(points[1].y).toBe(0.3);
    expect(points[0].x).toBeLessThan(points[1].x);
  });

  it("assigns each series a distinct color in first-appearance order", () => {
    const series = buildSeries([
      entry("b", 0, 0.1, "2026-01-01T00:00:00Z"),
      entry("a", 0, 0.2, "2026-01-01T00:00:00Z"),
    ]);
    expect(series.find((s) => s.name === "b")?.color).toBe(seriesColor(0));
    expect(series.find((s) => s.name === "a")?.color).toBe(seriesColor(1));
  });

  it("returns an empty list for empty history", () => {
    expect(buildSeries([])).toEqual([]);
  });
});

describe("seriesColor", () => {
  it("never repeats a hue across the first 8 slots", () => {
    const colors = Array.from({ length: 8 }, (_, i) => seriesColor(i));
    expect(new Set(colors).size).toBe(8);
  });

  it("folds anything past the 8th slot into one muted 'Other' color", () => {
    expect(seriesColor(8)).toBe(seriesColor(9));
    expect(seriesColor(8)).not.toBe(seriesColor(7));
  });
});
