import type { PipelineNode } from "../types";

export interface NodeGroup {
  key: string;
  parent: PipelineNode;
  children: PipelineNode[];
}

export interface GroupedGraph {
  groups: NodeGroup[];
  edges: [string, string][];
}

// Turns the flat node/edge list into render-level groups: an agent that owns
// two or more leaf dependents (non-agent, single inbound edge, no outbound
// edges of their own) gets them folded into one composite card -- the
// "Radis-Master + its instances" pattern. A lone dependent, a fan-in target
// shared by more than one parent, or a dependent that has downstream edges
// of its own stays a first-class card so the graph never hides real
// structure behind a group. Grouping only ever pulls from directly declared
// edges, so absorbed children disappear from the returned edge list (the
// nesting already shows that relationship) while every other edge passes
// through untouched.
export function groupPipelineNodes(nodes: PipelineNode[], edges: [string, string][]): GroupedGraph {
  const byName = new Map(nodes.map((n) => [n.name, n]));
  const inDegree = new Map<string, number>();
  const outDegree = new Map<string, number>();
  for (const n of nodes) {
    inDegree.set(n.name, 0);
    outDegree.set(n.name, 0);
  }
  for (const [from, to] of edges) {
    if (!byName.has(from) || !byName.has(to)) continue;
    outDegree.set(from, (outDegree.get(from) ?? 0) + 1);
    inDegree.set(to, (inDegree.get(to) ?? 0) + 1);
  }

  const candidatesByParent = new Map<string, PipelineNode[]>();
  for (const [from, to] of edges) {
    const parent = byName.get(from);
    const child = byName.get(to);
    if (!parent || !child) continue;
    if (parent.type !== "agent") continue;
    if (child.type === "agent") continue;
    if (inDegree.get(to) !== 1) continue;
    if (outDegree.get(to) !== 0) continue;
    if (!candidatesByParent.has(from)) candidatesByParent.set(from, []);
    candidatesByParent.get(from)!.push(child);
  }

  const groupChildren = new Map<string, PipelineNode[]>();
  for (const [parentName, kids] of candidatesByParent) {
    if (kids.length >= 2) groupChildren.set(parentName, kids);
  }

  const absorbed = new Set<string>();
  for (const kids of groupChildren.values()) {
    for (const k of kids) absorbed.add(k.name);
  }

  const groups: NodeGroup[] = nodes
    .filter((n) => !absorbed.has(n.name))
    .map((n) => ({ key: n.name, parent: n, children: groupChildren.get(n.name) ?? [] }));

  const groupEdges = edges.filter(([from, to]) => !absorbed.has(from) && !absorbed.has(to));

  return { groups, edges: groupEdges };
}
