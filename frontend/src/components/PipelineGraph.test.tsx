import { fireEvent, render, screen } from "@testing-library/react";
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

  it("groups an agent's two-or-more leaf dependents into one composite card, with no separate edge lines to them", () => {
    render(
      <PipelineGraph
        nodes={[
          { name: "orchestrator", type: "agent", risk_score: 0.4 },
          { name: "search", type: "tool", risk_score: 0.1 },
          { name: "embeddings", type: "model_endpoint", risk_score: 0.2 },
        ]}
        edges={[
          ["orchestrator", "search"],
          ["orchestrator", "embeddings"],
        ]}
      />,
    );

    expect(screen.getByTestId("node-orchestrator")).toBeInTheDocument();
    expect(screen.getByTestId("node-search")).toBeInTheDocument();
    expect(screen.getByTestId("node-embeddings")).toBeInTheDocument();
    expect(screen.queryByTestId("edge-orchestrator-search")).not.toBeInTheDocument();
    expect(screen.queryByTestId("edge-orchestrator-embeddings")).not.toBeInTheDocument();
  });

  it("collapses and re-expands a composite card's children on header click", () => {
    render(
      <PipelineGraph
        nodes={[
          { name: "orchestrator", type: "agent", risk_score: 0.4 },
          { name: "search", type: "tool", risk_score: 0.1 },
          { name: "embeddings", type: "model_endpoint", risk_score: 0.2 },
        ]}
        edges={[
          ["orchestrator", "search"],
          ["orchestrator", "embeddings"],
        ]}
      />,
    );

    const toggle = screen.getByRole("button", { name: /collapse orchestrator/i });
    expect(screen.getByTestId("node-search")).toBeInTheDocument();

    fireEvent.click(toggle);
    expect(screen.queryByTestId("node-search")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /expand orchestrator/i }));
    expect(screen.getByTestId("node-search")).toBeInTheDocument();
  });
});
