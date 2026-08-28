export type NodeType = "agent" | "tool" | "model_endpoint" | "vector_store";

export interface PipelineNode {
  name: string;
  type: NodeType;
  risk_score: number | null;
}

export interface PipelineView {
  run_id: string;
  step: number;
  nodes: PipelineNode[];
  edges: [string, string][];
}

export interface ScoreHistoryEntry {
  step: number;
  node_name: string;
  risk_score: number;
  predicted_at: string;
}

export interface IncidentEntry {
  node_name: string;
  incident_type: string;
  occurred_at: string;
  source: string;
}

export interface TrackRecordView {
  run_id: string;
  history: ScoreHistoryEntry[];
  incidents: IncidentEntry[];
}
