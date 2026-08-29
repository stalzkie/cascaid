import type { ScoreHistoryEntry } from "../types";

// Fixed categorical order (never cycled/generated) -- see dataviz skill's
// reference palette. Each slot is a CSS custom property (index.css) so the
// dark-mode-stepped hue swaps in automatically instead of the line staying
// pinned to its light-mode hex. An entity past the 8th falls into one shared
// "Other" gray rather than a repeated or invented hue.
const CATEGORICAL = [
  "var(--series-1)", // blue
  "var(--series-2)", // orange
  "var(--series-3)", // aqua
  "var(--series-4)", // yellow
  "var(--series-5)", // magenta
  "var(--series-6)", // green
  "var(--series-7)", // violet
  "var(--series-8)", // red
];
const OTHER = "var(--muted)";

export function seriesColor(index: number): string {
  return CATEGORICAL[index] ?? OTHER;
}

export interface SeriesPoint {
  x: number; // ms since epoch
  y: number; // risk score
}

export interface Series {
  name: string;
  color: string;
  points: SeriesPoint[];
}

export function buildSeries(history: ScoreHistoryEntry[]): Series[] {
  const order: string[] = [];
  const byNode = new Map<string, SeriesPoint[]>();
  for (const entry of history) {
    if (!byNode.has(entry.node_name)) {
      byNode.set(entry.node_name, []);
      order.push(entry.node_name);
    }
    byNode.get(entry.node_name)!.push({ x: Date.parse(entry.predicted_at), y: entry.risk_score });
  }
  return order.map((name, i) => ({
    name,
    color: seriesColor(i),
    points: [...byNode.get(name)!].sort((a, b) => a.x - b.x),
  }));
}
