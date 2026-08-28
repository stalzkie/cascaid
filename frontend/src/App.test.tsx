import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("App", () => {
  it("loads and renders the pipeline graph and track record for an entered run id", async () => {
    const pipelineBody = {
      run_id: "run-1",
      step: 0,
      nodes: [{ name: "agent", type: "agent", risk_score: 0.2 }],
      edges: [],
    };
    const trackRecordBody = { run_id: "run-1", history: [], incidents: [] };
    globalThis.fetch = vi.fn((url: string) => {
      const body = url.includes("/pipeline/") ? pipelineBody : trackRecordBody;
      return Promise.resolve({ ok: true, status: 200, json: async () => body });
    }) as unknown as typeof fetch;

    render(<App />);
    fireEvent.change(screen.getByLabelText(/run id/i), { target: { value: "run-1" } });
    fireEvent.click(screen.getByRole("button", { name: /load/i }));

    await waitFor(() => expect(screen.getByRole("img", { name: /pipeline risk graph/i })).toBeInTheDocument());
    expect(screen.getByText(/no risk history/i)).toBeInTheDocument();
  });

  it("shows an error message when the API call fails", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network down")) as unknown as typeof fetch;

    render(<App />);
    fireEvent.change(screen.getByLabelText(/run id/i), { target: { value: "run-1" } });
    fireEvent.click(screen.getByRole("button", { name: /load/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/network down/i));
  });
});
