import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { NodeStatusTable } from "./NodeStatusTable";

describe("NodeStatusTable", () => {
  it("lists every node with its type, status, and score", () => {
    render(
      <NodeStatusTable
        nodes={[
          { name: "orchestrator", type: "agent", risk_score: 0.85 },
          { name: "search", type: "tool", risk_score: null },
        ]}
      />,
    );

    expect(screen.getByRole("table", { name: /node status/i })).toBeInTheDocument();
    expect(screen.getByText("orchestrator")).toBeInTheDocument();
    expect(screen.getByText("Critical risk")).toBeInTheDocument();
    expect(screen.getByText("0.85")).toBeInTheDocument();
    expect(screen.getByText("search")).toBeInTheDocument();
    expect(screen.getByText("Not yet scored")).toBeInTheDocument();
  });

  it("sorts by descending risk so the riskiest nodes lead", () => {
    render(
      <NodeStatusTable
        nodes={[
          { name: "low", type: "tool", risk_score: 0.1 },
          { name: "high", type: "tool", risk_score: 0.9 },
        ]}
      />,
    );

    const rows = screen.getAllByRole("row").slice(1); // drop header row
    expect(rows[0]).toHaveTextContent("high");
    expect(rows[1]).toHaveTextContent("low");
  });

  it("shows an empty note when there are no nodes yet", () => {
    render(<NodeStatusTable nodes={[]} />);
    expect(screen.getByRole("status")).toHaveTextContent(/no nodes/i);
  });
});
