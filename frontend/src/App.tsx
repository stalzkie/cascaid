import { useCallback, useEffect, useMemo, useState } from "react";
import { Login } from "./components/Login";
import { NodeStatusTable } from "./components/NodeStatusTable";
import { PipelineGraph } from "./components/PipelineGraph";
import { TrackRecordChart } from "./components/TrackRecordChart";
import { fetchPipeline, fetchRuns, fetchTrackRecord, UnauthorizedError } from "./lib/api";
import { getToken, logout } from "./lib/auth";
import { riskStatus, statusForBand, type RiskBand } from "./lib/riskStatus";
import type { PipelineView, TrackRecordView } from "./types";

const STATUS_BANDS: RiskBand[] = ["critical", "serious", "warning", "good"];

const API_BASE_URL = import.meta.env.VITE_DASHBOARD_API_URL ?? "http://localhost:8001";
const REFRESH_INTERVAL_MS = 10_000;

type LoadState = { status: "idle" } | { status: "loading" } | { status: "error"; message: string } | { status: "loaded" };

export function App() {
  const [authed, setAuthed] = useState(() => getToken() !== null);
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
    } catch (err) {
      if (err instanceof UnauthorizedError) setAuthed(false);
      // Otherwise non-fatal: the run picker just stays at whatever it last had.
    }
  }, []);

  useEffect(() => {
    if (!authed) return;
    void refreshRuns();
    const interval = setInterval(() => void refreshRuns(), REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [authed, refreshRuns]);

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
      if (err instanceof UnauthorizedError) {
        setAuthed(false);
        return;
      }
      setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  useEffect(() => {
    if (!authed || !runId || !autoRefresh) return;
    const interval = setInterval(() => void load(runId), REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [authed, runId, autoRefresh, load]);

  const statusCounts = useMemo(() => {
    const counts = new Map<RiskBand, number>();
    for (const node of pipeline?.nodes ?? []) {
      const band = riskStatus(node.risk_score).band;
      counts.set(band, (counts.get(band) ?? 0) + 1);
    }
    return counts;
  }, [pipeline]);

  // The same box used to jump straight to an id also filters the list below
  // it live -- one control instead of a separate search + browse pair, and
  // it's what keeps a long run history from becoming an unbounded wall of
  // pills (see .run-list's own max-height + scroll for the other half of
  // that fix).
  const filteredRuns = useMemo(() => {
    const q = runIdInput.trim().toLowerCase();
    if (!q) return knownRuns;
    return knownRuns.filter((id) => id.toLowerCase().includes(q));
  }, [knownRuns, runIdInput]);

  if (!authed) {
    return <Login apiBaseUrl={API_BASE_URL} onSuccess={() => setAuthed(true)} />;
  }

  return (
    <main className="app">
      <header className="app-header">
        <div>
          <span className="eyebrow">Cascaid</span>
          <h1>Risk Dashboard</h1>
        </div>
        {lastUpdated && state.status === "loaded" && (
          <p className="last-updated">Last updated: {lastUpdated.toLocaleTimeString()}</p>
        )}
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => void logout(API_BASE_URL).then(() => setAuthed(false))}
        >
          Log out
        </button>
      </header>

      <div className="run-picker">
        <form
          className="toolbar"
          onSubmit={(e) => {
            e.preventDefault();
            if (runIdInput.trim()) void load(runIdInput.trim());
          }}
        >
          <label htmlFor="run-id" className="sr-only">
            Run ID
          </label>
          <input
            id="run-id"
            type="text"
            value={runIdInput}
            onChange={(e) => setRunIdInput(e.target.value)}
            placeholder="Search or jump to a run id, e.g. run-1"
          />
          <button type="submit" className="btn btn-primary" disabled={state.status === "loading"}>
            Load
          </button>
          {runId && (
            <button type="button" className="btn btn-secondary" onClick={() => void load(runId)} disabled={state.status === "loading"}>
              Refresh
            </button>
          )}
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              className="toggle-switch"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
        </form>

        {knownRuns.length === 0 ? (
          <p role="status" className="empty-note">
            No runs found yet. If you just ran <code>docker compose up</code>, the seed step may still be finishing.
          </p>
        ) : (
          <ul className="run-list">
            {filteredRuns.length === 0 ? (
              <li className="run-list-empty">No runs match "{runIdInput.trim()}"</li>
            ) : (
              filteredRuns.map((id) => (
                <li key={id}>
                  <button type="button" className={id === runId ? "run-list-item active" : "run-list-item"} onClick={() => void load(id)}>
                    {id}
                  </button>
                </li>
              ))
            )}
          </ul>
        )}
      </div>

      {state.status === "error" && <p role="alert">{state.message}</p>}

      {state.status === "loaded" && (
        <>
          {pipeline && pipeline.nodes.length > 0 && (
            <div className="status-strip">
              {STATUS_BANDS.map((band) => {
                const status = statusForBand(band);
                return (
                  <div className="status-tile" key={band}>
                    <span className="status-tile-dot" style={{ color: status.color }} />
                    <div>
                      <p className="status-tile-count">{statusCounts.get(band) ?? 0}</p>
                      <p className="status-tile-label">{status.label}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <section className="card">
            <p className="section-eyebrow">Pipeline</p>
            {pipeline ? (
              <PipelineGraph nodes={pipeline.nodes} edges={pipeline.edges} />
            ) : (
              <p role="status" className="empty-note">
                No snapshot has been ingested for this run yet.
              </p>
            )}
          </section>
          <div className="bottom-row">
            <section className="card">
              <p className="section-eyebrow">Track record</p>
              <TrackRecordChart history={trackRecord?.history ?? []} incidents={trackRecord?.incidents ?? []} />
            </section>
            <section className="card">
              <p className="section-eyebrow">Node status</p>
              <NodeStatusTable nodes={pipeline?.nodes ?? []} />
            </section>
          </div>
        </>
      )}
    </main>
  );
}
