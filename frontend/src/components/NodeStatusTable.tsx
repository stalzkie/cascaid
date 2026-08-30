import type { CSSProperties } from "react";
import { nodeTypeColorVar, nodeTypeLabel } from "../lib/nodeTypeStyle";
import { riskStatus } from "../lib/riskStatus";
import type { PipelineNode } from "../types";

interface Props {
  nodes: PipelineNode[];
}

// The graph shows structure; this shows priority -- riskiest first, plain
// rows, so "what needs attention right now" doesn't require reading the
// canvas layout. Score, not name, drives order for exactly that reason.
export function NodeStatusTable({ nodes }: Props) {
  if (nodes.length === 0) {
    return (
      <p role="status" className="empty-note">
        No nodes to show yet for this run.
      </p>
    );
  }

  const sorted = [...nodes].sort((a, b) => (b.risk_score ?? -1) - (a.risk_score ?? -1));

  return (
    <table className="node-status-table" aria-label="Node status">
      <thead>
        <tr>
          <th scope="col">Node</th>
          <th scope="col">Type</th>
          <th scope="col">Status</th>
          <th scope="col">Score</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((node) => {
          const status = riskStatus(node.risk_score);
          const scoreText = node.risk_score === null ? "—" : node.risk_score.toFixed(2);
          return (
            <tr key={node.name}>
              <td>
                <span className="node-status-table-name">
                  <span className="node-status-table-swatch" style={{ background: nodeTypeColorVar(node.type) }} />
                  {node.name}
                </span>
              </td>
              <td className="node-status-table-muted">{nodeTypeLabel(node.type)}</td>
              <td>
                <span
                  className="node-status-pill"
                  style={{ color: status.color, "--pill-color": status.color } as CSSProperties}
                >
                  <span className="node-status-pill-dot" style={{ background: status.color }} />
                  {status.label}
                </span>
              </td>
              <td className="node-status-table-score">{scoreText}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
