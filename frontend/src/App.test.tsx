import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function mockFetch({
  runIds = [] as string[],
  pipelineBody,
  trackRecordBody,
}: {
  runIds?: string[];
  pipelineBody?: unknown;
  trackRecordBody?: unknown;
}) {
  globalThis.fetch = vi.fn((url: string) => {
    if (url.includes("/runs")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ run_ids: runIds }) });
    }
    const body = url.includes("/pipeline/") ? pipelineBody : trackRecordBody;
    return Promise.resolve({ ok: true, status: 200, json: async () => body });
  }) as unknown as typeof fetch;
}

describe("App", () => {
  it("loads and renders the pipeline graph and track record for an entered run id", async () => {
    mockFetch({
      pipelineBody: {
        run_id: "run-1",
        step: 0,
        nodes: [{ name: "agent", type: "agent", risk_score: 0.2 }],
        edges: [],
      },
      trackRecordBody: { run_id: "run-1", history: [], incidents: [] },
    });

    render(<App />);
    fireEvent.change(screen.getByLabelText(/run id/i), { target: { value: "run-1" } });
    fireEvent.click(screen.getByRole("button", { name: /load/i }));

    await waitFor(() => expect(screen.getByRole("img", { name: /pipeline risk graph/i })).toBeInTheDocument());
    expect(screen.getByText(/no risk history/i)).toBeInTheDocument();
    expect(screen.getByText(/last updated/i)).toBeInTheDocument();
  });

  it("shows an error message when the API call fails", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network down")) as unknown as typeof fetch;

    render(<App />);
    fireEvent.change(screen.getByLabelText(/run id/i), { target: { value: "run-1" } });
    fireEvent.click(screen.getByRole("button", { name: /load/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/network down/i));
  });

  it("shows an empty-state message when no runs are known yet", async () => {
    mockFetch({ runIds: [] });

    render(<App />);

    await waitFor(() => expect(screen.getByText(/no runs found yet/i)).toBeInTheDocument());
  });

  it("lists known runs and loads one when clicked", async () => {
    mockFetch({
      runIds: ["run-a", "run-b"],
      pipelineBody: { run_id: "run-b", step: 0, nodes: [], edges: [] },
      trackRecordBody: { run_id: "run-b", history: [], incidents: [] },
    });

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: "run-b" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "run-b" }));

    await waitFor(() => expect(screen.getByRole("img", { name: /pipeline risk graph/i })).toBeInTheDocument());
  });
});
