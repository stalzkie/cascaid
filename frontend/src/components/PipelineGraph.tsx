import { useState } from "react";
import { computeLayout } from "../lib/layout";
import { groupPipelineNodes, type NodeGroup } from "../lib/nodeGroups";
import { nodeShape, type Shape } from "../lib/nodeShape";
import { nodeTypeColorVar, nodeTypeLabel } from "../lib/nodeTypeStyle";
import { riskStatus } from "../lib/riskStatus";
import type { NodeType, PipelineNode } from "../types";

interface Props {
  nodes: PipelineNode[];
  edges: [string, string][];
}

// Card geometry. Two independent channels are drawn on every card,
// deliberately never sharing a hue (see nodeTypeStyle.ts):
//   - IDENTITY (what kind of node this is) -> solid badge fill + filled glyph
//     + a small type caption, all in the type's color.
//   - STATUS (how worried to be about it) -> the card's own border color +
//     a status dot + a status word. Never color alone for either channel --
//     both carry a text label too.
//
// A group with >=2 leaf dependents (see nodeGroups.ts) renders as one
// composite card instead of separate cards + edges -- module shape follows
// module role: an orchestrator fanning out to tools/models looks visibly
// different from a lone node, without needing a different node type to
// exist. Collapsing a composite card's children is purely a view toggle;
// the underlying groups/layout are unaffected by which are collapsed except
// for the height each contributes to its layer.
const CARD_W = 216;
const LEAF_CARD_H = 84;
const HEADER_H = 64;
const CHILD_ROW_H = 34;
const CARD_PAD = 14;
const BADGE_SIZE = 34;
const CHILD_BADGE_SIZE = 22;
const MARGIN = 40;
const ROW_GAP = 36;

function typeGlyph(shape: Shape, cx: number, cy: number, r: number, fill: string) {
  switch (shape) {
    case "circle":
      return <circle cx={cx} cy={cy} r={r} fill={fill} />;
    case "square":
      return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx={3.5} fill={fill} />;
    case "diamond":
      return <polygon points={`${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`} fill={fill} />;
    case "hexagon": {
      const pts = Array.from({ length: 6 }, (_, i) => {
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
      }).join(" ");
      return <polygon points={pts} fill={fill} />;
    }
  }
}

function cardHeight(group: NodeGroup, collapsed: boolean): number {
  if (group.children.length === 0) return LEAF_CARD_H;
  if (collapsed) return HEADER_H;
  return HEADER_H + group.children.length * CHILD_ROW_H + CARD_PAD / 2;
}

export function PipelineGraph({ nodes, edges }: Props) {
  const [collapsedKeys, setCollapsedKeys] = useState<Set<string>>(new Set());
  const { groups, edges: groupEdges } = groupPipelineNodes(nodes, edges);
  const groupsByKey = new Map(groups.map((g) => [g.key, g]));

  const positions = computeLayout(
    groups.map((g) => g.key),
    groupEdges,
    {
      xSpacing: CARD_W + 88,
      ySpacing: ROW_GAP,
      nodeHeight: (key) => cardHeight(groupsByKey.get(key)!, collapsedKeys.has(key)),
    },
  );

  const heightAt = (key: string) => cardHeight(groupsByKey.get(key)!, collapsedKeys.has(key));
  const xs = Object.values(positions).map((p) => p.x);
  const tops = groups.map((g) => positions[g.key].y - heightAt(g.key) / 2);
  const bottoms = groups.map((g) => positions[g.key].y + heightAt(g.key) / 2);
  const minX = Math.min(0, ...xs);
  const maxX = Math.max(0, ...xs);
  const minY = Math.min(0, ...tops);
  const maxY = Math.max(0, ...bottoms);
  const width = maxX - minX + CARD_W + MARGIN * 2;
  const height = maxY - minY + MARGIN * 2;
  const ox = MARGIN + CARD_W / 2 - minX;
  const oy = MARGIN - minY;

  const byName = new Map(nodes.map((n) => [n.name, n]));
  const typesPresent = Array.from(new Set(nodes.map((n) => n.type))) as NodeType[];

  const toggleCollapsed = (key: string) => {
    setCollapsedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <div>
      {typesPresent.length > 0 && (
        <ul className="pg-legend">
          {typesPresent.map((type) => (
            <li key={type} className="pg-legend-item">
              <span className="pg-legend-swatch" style={{ background: nodeTypeColorVar(type) }} />
              {nodeTypeLabel(type)}
            </li>
          ))}
        </ul>
      )}
      <svg
        role="img"
        aria-label="Pipeline risk graph"
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ minHeight: 300, display: "block" }}
      >
        <defs>
          <pattern id="pg-dot-grid" width={22} height={22} patternUnits="userSpaceOnUse">
            <circle cx={1.5} cy={1.5} r={1.5} fill="var(--pg-dot)" />
          </pattern>
        </defs>
        <rect x={0} y={0} width={width} height={height} rx={8} fill="var(--pg-canvas)" stroke="var(--pg-node-border)" strokeWidth={1} />
        <rect x={0} y={0} width={width} height={height} rx={8} fill="url(#pg-dot-grid)" />

        <g>
          {groupEdges.map(([from, to]) => {
            const a = positions[from];
            const b = positions[to];
            if (!a || !b) return null;
            const target = byName.get(to);
            const targetStatus = riskStatus(target?.risk_score ?? null);
            const sx = a.x + ox + CARD_W / 2;
            const sy = a.y + oy;
            const tx = b.x + ox - CARD_W / 2;
            const ty = b.y + oy;
            const midX = sx + (tx - sx) / 2;
            const path = `M ${sx},${sy} C ${midX},${sy} ${midX},${ty} ${tx},${ty}`;
            return (
              <g key={`${from}->${to}`}>
                <path
                  data-testid={`edge-${from}-${to}`}
                  d={path}
                  fill="none"
                  stroke="var(--pg-edge)"
                  strokeWidth={2}
                  strokeLinecap="round"
                />
                <circle cx={tx} cy={ty} r={3.5} fill={targetStatus.color} />
              </g>
            );
          })}
        </g>

        <g>
          {groups.map((group) => {
            const pos = positions[group.key];
            if (!pos) return null;
            const collapsed = collapsedKeys.has(group.key);
            const h = cardHeight(group, collapsed);
            const status = riskStatus(group.parent.risk_score);
            const typeColor = nodeTypeColorVar(group.parent.type);
            const cx = pos.x + ox;
            const cy = pos.y + oy;
            const cardX = cx - CARD_W / 2;
            const cardY = cy - h / 2;
            const scoreText = group.parent.risk_score === null ? "—" : group.parent.risk_score.toFixed(2);
            const hasChildren = group.children.length > 0;
            // Leaf cards keep the full-height layout (badge+name+type on the
            // left, score top-right, status dot+word bottom-left) with room
            // to breathe. A composite header is shorter, so its right column
            // mirrors the left one line-for-line (score above status word,
            // same two y-coordinates as name above type) instead of trying
            // to fit a bottom status row too -- and the collapse toggle gets
            // its own row above both columns so it never fights either for
            // space.
            const badgeCx = cardX + CARD_PAD + BADGE_SIZE / 2;
            const badgeCy = hasChildren ? cardY + 40 : cardY + HEADER_H / 2 + 2;
            const textX = cardX + CARD_PAD + BADGE_SIZE + 10;
            const rightX = cardX + CARD_W - CARD_PAD;
            const toggleCx = cardX + CARD_W - CARD_PAD - 9;
            const toggleCy = cardY + 15;

            return (
              <g key={group.key} data-testid={`node-${group.parent.name}`}>
                <rect
                  x={cardX}
                  y={cardY}
                  width={CARD_W}
                  height={h}
                  rx={6}
                  fill="var(--pg-node-bg)"
                  stroke={status.color}
                  strokeWidth={1.5}
                />
                {hasChildren && !collapsed && (
                  <line
                    x1={cardX + CARD_PAD}
                    y1={cardY + HEADER_H}
                    x2={cardX + CARD_W - CARD_PAD}
                    y2={cardY + HEADER_H}
                    stroke="var(--pg-node-border)"
                    strokeWidth={1}
                  />
                )}

                <rect x={badgeCx - BADGE_SIZE / 2} y={badgeCy - BADGE_SIZE / 2} width={BADGE_SIZE} height={BADGE_SIZE} rx={6} fill={typeColor} />
                {typeGlyph(nodeShape(group.parent.type), badgeCx, badgeCy, 7.5, "var(--node-ink)")}

                <text x={textX} y={badgeCy - 5} textAnchor="start" fontSize={13} fontWeight={600} fill="var(--text-primary)">
                  {group.parent.name}
                </text>
                <text
                  x={textX}
                  y={badgeCy + 10}
                  textAnchor="start"
                  fontSize={9}
                  fontWeight={600}
                  letterSpacing={0.5}
                  fill={typeColor}
                  style={{ textTransform: "uppercase" }}
                >
                  {nodeTypeLabel(group.parent.type)}
                </text>

                {hasChildren ? (
                  <>
                    <text x={rightX} y={badgeCy - 5} textAnchor="end" fontSize={12.5} fontWeight={600} fill="var(--text-primary)">
                      {scoreText}
                    </text>
                    <text x={rightX} y={badgeCy + 10} textAnchor="end" fontSize={9.5} fontWeight={600} fill={status.color}>
                      {status.label}
                    </text>
                  </>
                ) : (
                  <>
                    <text x={rightX} y={cardY + 20} textAnchor="end" fontSize={12.5} fontWeight={600} fill="var(--text-primary)">
                      {scoreText}
                    </text>
                    <circle cx={cardX + CARD_PAD + 3.5} cy={cardY + h - 14} r={3.5} fill={status.color} />
                    <text x={cardX + CARD_PAD + 13} y={cardY + h - 10} textAnchor="start" fontSize={10.5} fill="var(--pg-text)">
                      {status.label}
                    </text>
                  </>
                )}

                {hasChildren && (
                  <g
                    className="pg-collapse-toggle"
                    role="button"
                    tabIndex={0}
                    aria-label={`${collapsed ? "Expand" : "Collapse"} ${group.parent.name}`}
                    onClick={() => toggleCollapsed(group.key)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") toggleCollapsed(group.key);
                    }}
                  >
                    <circle cx={toggleCx} cy={toggleCy} r={7} fill="var(--surface-muted)" stroke="var(--border)" />
                    <line x1={toggleCx - 3.5} y1={toggleCy} x2={toggleCx + 3.5} y2={toggleCy} stroke="var(--text-secondary)" strokeWidth={1.5} />
                    {collapsed && <line x1={toggleCx} y1={toggleCy - 3.5} x2={toggleCx} y2={toggleCy + 3.5} stroke="var(--text-secondary)" strokeWidth={1.5} />}
                  </g>
                )}

                {hasChildren && !collapsed && (
                  <g>
                    {group.children.map((child, i) => {
                      const rowY = cardY + HEADER_H + i * CHILD_ROW_H + CHILD_ROW_H / 2;
                      const childStatus = riskStatus(child.risk_score);
                      const childColor = nodeTypeColorVar(child.type);
                      const childScore = child.risk_score === null ? "—" : child.risk_score.toFixed(2);
                      const badgeX = cardX + CARD_PAD + CHILD_BADGE_SIZE / 2;
                      return (
                        <g key={child.name} data-testid={`node-${child.name}`}>
                          <rect
                            x={badgeX - CHILD_BADGE_SIZE / 2}
                            y={rowY - CHILD_BADGE_SIZE / 2}
                            width={CHILD_BADGE_SIZE}
                            height={CHILD_BADGE_SIZE}
                            rx={4}
                            fill={childColor}
                            opacity={0.85}
                          />
                          {typeGlyph(nodeShape(child.type), badgeX, rowY, 5, "var(--node-ink)")}
                          <circle cx={badgeX + CHILD_BADGE_SIZE / 2 + 9} cy={rowY - 8} r={3} fill={childStatus.color} />
                          <text x={badgeX + CHILD_BADGE_SIZE / 2 + 8} y={rowY + 4} textAnchor="start" fontSize={11.5} fontWeight={600} fill="var(--text-primary)">
                            {child.name}
                          </text>
                          <text x={cardX + CARD_W - CARD_PAD} y={rowY + 4} textAnchor="end" fontSize={11} fill="var(--pg-text)">
                            {childScore}
                          </text>
                        </g>
                      );
                    })}
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
