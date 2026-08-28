import type { PipelineView, TrackRecordView } from "../types";

export async function fetchPipeline(apiBaseUrl: string, runId: string): Promise<PipelineView | null> {
  const response = await fetch(`${apiBaseUrl}/pipeline/${runId}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`GET /pipeline/${runId} failed: ${response.status}`);
  return response.json() as Promise<PipelineView>;
}

export async function fetchTrackRecord(apiBaseUrl: string, runId: string): Promise<TrackRecordView> {
  const response = await fetch(`${apiBaseUrl}/track-record/${runId}`);
  if (!response.ok) throw new Error(`GET /track-record/${runId} failed: ${response.status}`);
  return response.json() as Promise<TrackRecordView>;
}
