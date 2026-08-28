import { describe, expect, it } from "vitest";
import { scaleLinear } from "./scale";

describe("scaleLinear", () => {
  it("maps domain endpoints to range endpoints", () => {
    const scale = scaleLinear([0, 10], [0, 100]);
    expect(scale(0)).toBe(0);
    expect(scale(10)).toBe(100);
    expect(scale(5)).toBe(50);
  });

  it("supports an inverted range", () => {
    const scale = scaleLinear([0, 10], [100, 0]);
    expect(scale(0)).toBe(100);
    expect(scale(10)).toBe(0);
  });

  it("returns the midpoint of the range when the domain has zero span", () => {
    const scale = scaleLinear([5, 5], [0, 100]);
    expect(scale(5)).toBe(50);
  });
});
