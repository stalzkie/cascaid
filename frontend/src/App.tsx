import { useState } from "react";
import { PipelineGraph } from "./components/PipelineGraph";
import { TrackRecordChart } from "./components/TrackRecordChart";
import { fetchPipeline, fetchTrackRecord } from "./lib/api";
import type { PipelineView, TrackRecordView } from "./types";

const API_BASE_URL = import.meta.env.VITE_DASHBOARD_API_URL ?? "http://localhost:8001";

type LoadState = { status: "idle" } | { status: "loading" } | { status: "error"; message: string } | { status: "loaded" };

export function App() {
  const [runIdInput, setRunIdInput] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [pipeline, setPipeline] = useState<PipelineView | null>(null);
  const [trackRecord, setTrackRecord] = useState<TrackRecordView | null>(null);
  const [state, setState] = useState<LoadState>({ status: "idle" });

  async function load(id: string) {
    setState({ status: "loading" });
    try {
      const [pipelineView, trackRecordView] = await Promise.all([
        fetchPipeline(API_BASE_URL, id),
        fetchTrackRecord(API_BASE_URL, id),
      ]);
      setRunId(id);
      setPipeline(pipelineView);
      setTrackRecord(trackRecordView);
      setState({ status: "loaded" });
    } catch (err) {
      setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  return (
    <main className="app">
      <h1>Cascaid — Risk Dashboard</h1>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (runIdInput.trim()) void load(runIdInput.trim());
        }}
      >
        <label htmlFor="run-id">Run ID</label>
        <input id="run-id" value={runIdInput} onChange={(e) => setRunIdInput(e.target.value)} placeholder="e.g. run-1" />
        <button type="submit" disabled={state.status === "loading"}>
          Load
        </button>
        {runId && (
          <button type="button" onClick={() => void load(runId)} disabled={state.status === "loading"}>
            Refresh
          </button>
        )}
      </form>

      {state.status === "error" && <p role="alert">{state.message}</p>}

      {state.status === "loaded" && (
        <>
          <section>
            <h2>Pipeline</h2>
            {pipeline ? (
              <PipelineGraph nodes={pipeline.nodes} edges={pipeline.edges} />
            ) : (
              <p role="status">No snapshot has been ingested for this run yet.</p>
            )}
          </section>
          <section>
            <h2>Track record</h2>
            <TrackRecordChart history={trackRecord?.history ?? []} incidents={trackRecord?.incidents ?? []} />
          </section>
        </>
      )}
    </main>
  );
}
