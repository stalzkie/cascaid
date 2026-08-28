import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrackRecordChart } from "./TrackRecordChart";

describe("TrackRecordChart", () => {
  it("shows an empty-state message when there is no history yet", () => {
    render(<TrackRecordChart history={[]} incidents={[]} />);

    expect(screen.getByRole("status")).toHaveTextContent(/no risk history/i);
  });

  it("renders one line per node and one marker per incident", () => {
    render(
      <TrackRecordChart
        history={[
          { step: 0, node_name: "agent", risk_score: 0.1, predicted_at: "2026-01-01T00:00:00Z" },
          { step: 1, node_name: "agent", risk_score: 0.4, predicted_at: "2026-01-01T00:01:00Z" },
          { step: 0, node_name: "store", risk_score: 0.2, predicted_at: "2026-01-01T00:00:00Z" },
        ]}
        incidents={[{ node_name: "agent", incident_type: "degradation", occurred_at: "2026-01-01T00:00:30Z", source: "manual" }]}
      />,
    );

    expect(screen.getByTestId("series-agent")).toBeInTheDocument();
    expect(screen.getByTestId("series-store")).toBeInTheDocument();
    expect(screen.getAllByTestId("incident-marker")).toHaveLength(1);
    expect(screen.getByText("agent")).toBeInTheDocument();
    expect(screen.getByText("Incident")).toBeInTheDocument();
  });
});
