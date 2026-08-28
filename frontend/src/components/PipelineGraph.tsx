import { computeLayout } from "../lib/layout";
import { nodeShape, type Shape } from "../lib/nodeShape";
import { riskStatus } from "../lib/riskStatus";
import type { PipelineNode } from "../types";

interface Props {
  nodes: PipelineNode[];
  edges: [string, string][];
}

const NODE_RADIUS = 22;
const MARGIN = 60;

function shapeElement(shape: Shape, cx: number, cy: number, r: number, fill: string, name: string) {
  const common = { fill, stroke: "var(--pg-node-stroke)", strokeWidth: 2, "data-testid": `node-${name}` };
  switch (shape) {
    case "circle":
      return <circle cx={cx} cy={cy} r={r} {...common} />;
    case "square":
      return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={4} {...common} />;
    case "diamond":
      return (
        <polygon
          points={`${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`}
          {...common}
        />
      );
    case "hexagon": {
      const pts = Array.from({ length: 6 }, (_, i) => {
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
      }).join(" ");
      return <polygon points={pts} {...common} />;
    }
  }
}

export function PipelineGraph({ nodes, edges }: Props) {
  const positions = computeLayout(
    nodes.map((n) => n.name),
    edges,
  );
  const xs = Object.values(positions).map((p) => p.x);
  const ys = Object.values(positions).map((p) => p.y);
  const minX = Math.min(0, ...xs);
  const maxX = Math.max(0, ...xs);
  const minY = Math.min(0, ...ys);
  const maxY = Math.max(0, ...ys);
  const width = maxX - minX + MARGIN * 2;
  const height = maxY - minY + MARGIN * 2;
  const ox = MARGIN - minX;
  const oy = MARGIN - minY;

  return (
    <svg
      role="img"
      aria-label="Pipeline risk graph"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ minHeight: 240 }}
    >
      <g>
        {edges.map(([from, to]) => {
          const a = positions[from];
          const b = positions[to];
          if (!a || !b) return null;
          return (
            <line
              key={`${from}->${to}`}
              data-testid={`edge-${from}-${to}`}
              x1={a.x + ox}
              y1={a.y + oy}
              x2={b.x + ox}
              y2={b.y + oy}
              stroke="var(--pg-edge)"
              strokeWidth={2}
            />
          );
        })}
      </g>
      <g>
        {nodes.map((node) => {
          const pos = positions[node.name];
          if (!pos) return null;
          const status = riskStatus(node.risk_score);
          const cx = pos.x + ox;
          const cy = pos.y + oy;
          return (
            <g key={node.name}>
              {shapeElement(nodeShape(node.type), cx, cy, NODE_RADIUS, status.color, node.name)}
              <text x={cx} y={cy + NODE_RADIUS + 16} textAnchor="middle" fontSize={12} fill="var(--pg-text)">
                {node.name}
              </text>
              <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11} fill="#fff" fontWeight={600}>
                {node.risk_score === null ? "—" : node.risk_score.toFixed(2)}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
