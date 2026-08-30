import { buildSeries } from "../lib/series";
import { scaleLinear } from "../lib/scale";
import type { IncidentEntry, ScoreHistoryEntry } from "../types";

interface Props {
  history: ScoreHistoryEntry[];
  incidents: IncidentEntry[];
}

const WIDTH = 720;
const HEIGHT = 280;
const PAD = { top: 16, right: 16, bottom: 32, left: 40 };
const INCIDENT_COLOR = "#d03b3b"; // status: critical -- reserved, not a series hue

export function TrackRecordChart({ history, incidents }: Props) {
  const series = buildSeries(history);
  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;

  if (series.length === 0) {
    return (
      <p role="status" className="empty-note">
        No risk history recorded yet for this run.
      </p>
    );
  }

  const allX = series.flatMap((s) => s.points.map((p) => p.x)).concat(incidents.map((i) => Date.parse(i.occurred_at)));
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const x = scaleLinear([minX, maxX], [PAD.left, PAD.left + plotWidth]);
  const y = scaleLinear([0, 1], [PAD.top + plotHeight, PAD.top]);

  return (
    <div>
      <svg role="img" aria-label="Predicted risk history vs. actual incidents" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={PAD.left} y1={y(tick)} x2={WIDTH - PAD.right} y2={y(tick)} stroke="var(--trc-grid)" strokeWidth={1} />
            <text x={PAD.left - 8} y={y(tick) + 4} textAnchor="end" fontSize={10} fill="var(--trc-muted)">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        {incidents.map((incident, i) => {
          const ix = x(Date.parse(incident.occurred_at));
          return (
            <g key={`${incident.node_name}-${incident.occurred_at}-${i}`} data-testid="incident-marker">
              <line x1={ix} y1={PAD.top} x2={ix} y2={PAD.top + plotHeight} stroke={INCIDENT_COLOR} strokeWidth={1.5} strokeDasharray="4 3" />
              <title>{`Incident: ${incident.node_name} (${incident.incident_type}) at ${incident.occurred_at}`}</title>
            </g>
          );
        })}

        {series.map((s) => (
          <g key={s.name}>
            <polyline
              data-testid={`series-${s.name}`}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              points={s.points.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
            />
            {s.points.map((p, i) => (
              <circle key={i} cx={x(p.x)} cy={y(p.y)} r={3} fill={s.color}>
                <title>{`${s.name}: ${p.y.toFixed(2)}`}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <ul className="trc-legend">
        {series.map((s) => (
          <li key={s.name} className="trc-legend-item">
            <span className="trc-swatch" style={{ background: s.color }} />
            {s.name}
          </li>
        ))}
        {incidents.length > 0 && (
          <li className="trc-legend-item">
            <span className="trc-swatch" style={{ background: INCIDENT_COLOR }} />
            Incident
          </li>
        )}
      </ul>
    </div>
  );
}
