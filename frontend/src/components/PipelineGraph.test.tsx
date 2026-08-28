import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PipelineGraph } from "./PipelineGraph";

describe("PipelineGraph", () => {
  it("renders one shape per node and one line per edge", () => {
    render(
      <PipelineGraph
        nodes={[
          { name: "agent", type: "agent", risk_score: 0.9 },
          { name: "store", type: "vector_store", risk_score: null },
        ]}
        edges={[["agent", "store"]]}
      />,
    );

    expect(screen.getByTestId("node-agent")).toBeInTheDocument();
    expect(screen.getByTestId("node-store")).toBeInTheDocument();
    expect(screen.getByTestId("edge-agent-store")).toBeInTheDocument();
    expect(screen.getByText("0.90")).toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("renders the pipeline graph as an accessible image", () => {
    render(<PipelineGraph nodes={[{ name: "solo", type: "tool", risk_score: 0.1 }]} edges={[]} />);

    expect(screen.getByRole("img", { name: /pipeline risk graph/i })).toBeInTheDocument();
  });
});
