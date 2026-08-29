import type { PipelineView, TrackRecordView } from "../types";
import { clearToken, getToken } from "./auth";

export class UnauthorizedError extends Error {
  constructor() {
    super("session expired or invalid, please log in again");
    this.name = "UnauthorizedError";
  }
}

async function authFetch(url: string): Promise<Response> {
  const token = getToken();
  const response = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
  if (response.status === 401) {
    clearToken();
    throw new UnauthorizedError();
  }
  return response;
}

export async function fetchRuns(apiBaseUrl: string): Promise<string[]> {
  const response = await authFetch(`${apiBaseUrl}/runs`);
  if (!response.ok) throw new Error(`GET /runs failed: ${response.status}`);
  const body = (await response.json()) as { run_ids: string[] };
  return body.run_ids;
}

export async function fetchPipeline(apiBaseUrl: string, runId: string): Promise<PipelineView | null> {
  const response = await authFetch(`${apiBaseUrl}/pipeline/${runId}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`GET /pipeline/${runId} failed: ${response.status}`);
  return response.json() as Promise<PipelineView>;
}

export async function fetchTrackRecord(apiBaseUrl: string, runId: string): Promise<TrackRecordView> {
  const response = await authFetch(`${apiBaseUrl}/track-record/${runId}`);
  if (!response.ok) throw new Error(`GET /track-record/${runId} failed: ${response.status}`);
  return response.json() as Promise<TrackRecordView>;
}
