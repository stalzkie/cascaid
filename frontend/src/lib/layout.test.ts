import { describe, expect, it } from "vitest";
import { computeLayout } from "./layout";

describe("computeLayout", () => {
  it("places a single isolated node at the origin", () => {
    const pos = computeLayout(["a"], []);
    expect(pos.a).toEqual({ x: 0, y: 0 });
  });

  it("lays out a linear chain in increasing x, same layer -> same x", () => {
    const pos = computeLayout(
      ["a", "b", "c"],
      [
        ["a", "b"],
        ["b", "c"],
      ],
    );
    expect(pos.a.x).toBeLessThan(pos.b.x);
    expect(pos.b.x).toBeLessThan(pos.c.x);
    expect(pos.a.y).toBe(pos.b.y);
    expect(pos.b.y).toBe(pos.c.y);
  });

  it("splits a diamond into two nodes sharing one layer, vertically separated", () => {
    const pos = computeLayout(
      ["a", "b", "c", "d"],
      [
        ["a", "b"],
        ["a", "c"],
        ["b", "d"],
        ["c", "d"],
      ],
    );
    expect(pos.b.x).toBe(pos.c.x);
    expect(pos.b.y).not.toBe(pos.c.y);
    expect(pos.a.x).toBeLessThan(pos.b.x);
    expect(pos.d.x).toBeGreaterThan(pos.b.x);
  });

  it("terminates and positions every node even when the topology has a cycle", () => {
    const pos = computeLayout(
      ["a", "b"],
      [
        ["a", "b"],
        ["b", "a"],
      ],
    );
    expect(Object.keys(pos).sort()).toEqual(["a", "b"]);
  });

  it("positions every declared node even ones with no edges at all", () => {
    const pos = computeLayout(["a", "b", "c"], [["a", "b"]]);
    expect(Object.keys(pos).sort()).toEqual(["a", "b", "c"]);
  });

  it("stacks a layer by each node's own height, not a uniform slot", () => {
    // b is much taller than c -- the gap between their centers should grow
    // to match, instead of the fixed ySpacing a uniform-height layout would use.
    const pos = computeLayout(
      ["a", "b", "c"],
      [
        ["a", "b"],
        ["a", "c"],
      ],
      { ySpacing: 10, nodeHeight: (name) => (name === "b" ? 200 : 20) },
    );
    expect(pos.c.y - pos.b.y).toBeCloseTo((200 + 20) / 2 + 10);
  });

  it("keeps a single-node layer centered on the origin regardless of its height", () => {
    const pos = computeLayout(["a"], [], { nodeHeight: () => 500 });
    expect(pos.a).toEqual({ x: 0, y: 0 });
  });
});
