import type { NodeType } from "../types";

export type Shape = "circle" | "square" | "hexagon" | "diamond";

const SHAPE_BY_TYPE: Record<NodeType, Shape> = {
  agent: "circle",
  tool: "square",
  model_endpoint: "hexagon",
  vector_store: "diamond",
};

export function nodeShape(type: NodeType): Shape {
  return SHAPE_BY_TYPE[type];
}
