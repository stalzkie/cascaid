import { describe, expect, it } from "vitest";
import { nodeShape } from "./nodeShape";
import type { NodeType } from "../types";

describe("nodeShape", () => {
  it("gives every node type a distinct shape", () => {
    const types: NodeType[] = ["agent", "tool", "model_endpoint", "vector_store"];
    const shapes = new Set(types.map(nodeShape));
    expect(shapes.size).toBe(4);
  });

  it("is a pure lookup, not color -- type identity never depends on hue", () => {
    expect(nodeShape("agent")).toBe("circle");
    expect(nodeShape("tool")).toBe("square");
    expect(nodeShape("model_endpoint")).toBe("hexagon");
    expect(nodeShape("vector_store")).toBe("diamond");
  });
});
