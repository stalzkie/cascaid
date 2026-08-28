export interface Position {
  x: number;
  y: number;
}

interface LayoutOptions {
  xSpacing?: number;
  ySpacing?: number;
}

// Deterministic layered (Sugiyama-style) layout via Kahn's algorithm: nodes with
// no incoming edges start at layer 0, then each layer is "everything whose
// predecessors are all already placed". A cycle can leave nodes with no
// zero-indegree entry point -- those are dropped into the next layer as a group
// so layout always terminates and every declared node gets a position.
export function computeLayout(
  nodeNames: string[],
  edges: [string, string][],
  { xSpacing = 160, ySpacing = 90 }: LayoutOptions = {},
): Record<string, Position> {
  const outgoing = new Map<string, string[]>();
  const inDegree = new Map<string, number>();
  for (const name of nodeNames) {
    outgoing.set(name, []);
    inDegree.set(name, 0);
  }
  for (const [from, to] of edges) {
    if (!outgoing.has(from) || !inDegree.has(to)) continue;
    outgoing.get(from)!.push(to);
    inDegree.set(to, (inDegree.get(to) ?? 0) + 1);
  }

  const remaining = new Set(nodeNames);
  const layers: string[][] = [];
  let frontier = nodeNames.filter((n) => inDegree.get(n) === 0);

  while (remaining.size > 0) {
    if (frontier.length === 0) {
      // Cycle: nothing has zero indegree among what's left. Collapse the rest
      // into one final layer rather than looping forever.
      frontier = nodeNames.filter((n) => remaining.has(n));
    }
    layers.push(frontier);
    for (const name of frontier) remaining.delete(name);

    const next: string[] = [];
    for (const name of frontier) {
      for (const target of outgoing.get(name) ?? []) {
        if (!remaining.has(target)) continue;
        inDegree.set(target, (inDegree.get(target) ?? 1) - 1);
        if (inDegree.get(target) === 0) next.push(target);
      }
    }
    frontier = next;
  }

  const positions: Record<string, Position> = {};
  layers.forEach((layer, layerIndex) => {
    const offset = ((layer.length - 1) * ySpacing) / 2;
    layer.forEach((name, i) => {
      positions[name] = { x: layerIndex * xSpacing, y: i * ySpacing - offset };
    });
  });
  return positions;
}
