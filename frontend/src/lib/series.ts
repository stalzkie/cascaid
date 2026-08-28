import type { ScoreHistoryEntry } from "../types";

// Fixed categorical order (never cycled/generated) -- see dataviz skill's
// reference palette. An entity past the 8th falls into one shared "Other" gray
// rather than a repeated or invented hue.
const CATEGORICAL = [
  "#2a78d6", // blue
  "#eb6834", // orange
  "#1baf7a", // aqua
  "#eda100", // yellow
  "#e87ba4", // magenta
  "#008300", // green
  "#4a3aa7", // violet
  "#e34948", // red
];
const OTHER = "#898781";

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
