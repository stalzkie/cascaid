import { describe, expect, it } from "vitest";
import { riskStatus } from "./riskStatus";

describe("riskStatus", () => {
  it("returns the unscored band for null", () => {
    expect(riskStatus(null).band).toBe("unscored");
  });

  it("bands a low score as good", () => {
    expect(riskStatus(0.1).band).toBe("good");
  });

  it("bands a mid score as warning", () => {
    expect(riskStatus(0.4).band).toBe("warning");
  });

  it("bands a high score as serious", () => {
    expect(riskStatus(0.65).band).toBe("serious");
  });

  it("bands a very high score as critical", () => {
    expect(riskStatus(0.9).band).toBe("critical");
  });

  it("is inclusive at each threshold boundary", () => {
    expect(riskStatus(0.3).band).toBe("warning");
    expect(riskStatus(0.6).band).toBe("serious");
    expect(riskStatus(0.8).band).toBe("critical");
  });

  it("every band carries a distinct color and a human label", () => {
    const bands = [null, 0.1, 0.4, 0.65, 0.9].map((s) => riskStatus(s));
    const colors = new Set(bands.map((b) => b.color));
    expect(colors.size).toBe(5);
    for (const b of bands) {
      expect(b.label.length).toBeGreaterThan(0);
    }
  });
});
