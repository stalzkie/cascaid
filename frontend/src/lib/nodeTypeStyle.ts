import type { NodeType } from "../types";

// Identity color per node type -- deliberately a *different* channel from
// risk status (see riskStatus.ts): type is "what this node is", status is
// "how worried to be about it", and the two must never share a hue so a
// glance at a card can't confuse the two questions. vector_store gets no
// hue at all (var(--node-vector) is a neutral ink, not a categorical color)
// because a 4th hue that stays all-pairs CVD-safe *and* clear of the warm
// status ramp doesn't exist in the palette -- see index.css's comment above
// --node-agent and dataviz skill's palette.md. Its shape (diamond, see
// nodeShape.ts) carries identity alone here; that's the documented
// "composite encoding" fallback for a category past what color can hold.
const COLOR_VAR: Record<NodeType, string> = {
  agent: "var(--node-agent)",
  tool: "var(--node-tool)",
  model_endpoint: "var(--node-model)",
  vector_store: "var(--node-vector)",
};

const LABEL: Record<NodeType, string> = {
  agent: "Agent",
  tool: "Tool",
  model_endpoint: "Model endpoint",
  vector_store: "Vector store",
};

export function nodeTypeColorVar(type: NodeType): string {
  return COLOR_VAR[type];
}

export function nodeTypeLabel(type: NodeType): string {
  return LABEL[type];
}

export const NODE_TYPES: NodeType[] = ["agent", "tool", "model_endpoint", "vector_store"];
