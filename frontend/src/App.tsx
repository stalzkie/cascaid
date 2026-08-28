import { useCallback, useEffect, useState } from "react";
import { PipelineGraph } from "./components/PipelineGraph";
import { TrackRecordChart } from "./components/TrackRecordChart";
import { fetchPipeline, fetchRuns, fetchTrackRecord } from "./lib/api";
import type { PipelineView, TrackRecordView } from "./types";

const API_BASE_URL = import.meta.env.VITE_DASHBOARD_API_URL ?? "http://localhost:8001";
const REFRESH_INTERVAL_MS = 10_000;

type LoadState = { status: "idle" } | { status: "loading" } | { status: "error"; message: string } | { status: "loaded" };

export function App() {
  const [runIdInput, setRunIdInput] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [knownRuns, setKnownRuns] = useState<string[]>([]);
  const [pipeline, setPipeline] = useState<PipelineView | null>(null);
  const [trackRecord, setTrackRecord] = useState<TrackRecordView | null>(null);
  const [state, setState] = useState<LoadState>({ status: "idle" });
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const refreshRuns = useCallback(async () => {
    try {
      setKnownRuns(await fetchRuns(API_BASE_URL));
    } catch {
      // Non-fatal: the run picker just stays at whatever it last had.
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
    const interval = setInterval(() => void refreshRuns(), REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refreshRuns]);

  const load = useCallback(async (id: string) => {
    setState({ status: "loading" });
    try {
      const [pipelineView, trackRecordView] = await Promise.all([
        fetchPipeline(API_BASE_URL, id),
        fetchTrackRecord(API_BASE_URL, id),
      ]);
      setRunId(id);
      setPipeline(pipelineView);
      setTrackRecord(trackRecordView);
      setLastUpdated(new Date());
      setState({ status: "loaded" });
    } catch (err) {
      setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  useEffect(() => {
    if (!runId || !autoRefresh) return;
    const interval = setInterval(() => void load(runId), REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [runId, autoRefresh, load]);

  return (
    <main className="app">
      <h1>Cascaid — Risk Dashboard</h1>

      <section>
        <h2>Known runs</h2>
        {knownRuns.length === 0 ? (
          <p role="status">
            No runs found yet. If you just ran <code>docker compose up</code>, the seed step may still be finishing.
          </p>
        ) : (
          <ul className="run-list">
            {knownRuns.map((id) => (
              <li key={id}>
                <button type="button" className={id === runId ? "run-list-item active" : "run-list-item"} onClick={() => void load(id)}>
                  {id}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

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
        <label className="auto-refresh-toggle">
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
          Auto-refresh
        </label>
      </form>

      {state.status === "error" && <p role="alert">{state.message}</p>}

      {state.status === "loaded" && (
        <>
          {lastUpdated && (
            <p className="last-updated">Last updated: {lastUpdated.toLocaleTimeString()}</p>
          )}
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
