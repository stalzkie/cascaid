export interface Position {
  x: number;
  y: number;
}

interface LayoutOptions {
  xSpacing?: number;
  ySpacing?: number;
  nodeHeight?: (name: string) => number;
}

// Deterministic layered (Sugiyama-style) layout via Kahn's algorithm: nodes with
// no incoming edges start at layer 0, then each layer is "everything whose
// predecessors are all already placed". A cycle can leave nodes with no
// zero-indegree entry point -- those are dropped into the next layer as a group
// so layout always terminates and every declared node gets a position.
export function computeLayout(
  nodeNames: string[],
  edges: [string, string][],
  { xSpacing = 160, ySpacing = 90, nodeHeight = () => 0 }: LayoutOptions = {},
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

  // Each layer stacks its own nodes top-to-bottom by their actual height
  // (uniform ySpacing is just the gap between them) rather than dropping
  // every node into an equal-size slot, so a tall composite card doesn't
  // overlap its neighbors and short cards don't leave dead space around them.
  const positions: Record<string, Position> = {};
  layers.forEach((layer, layerIndex) => {
    const heights = layer.map(nodeHeight);
    const totalHeight = heights.reduce((sum, h) => sum + h, 0) + ySpacing * Math.max(0, layer.length - 1);
    let cursor = -totalHeight / 2;
    layer.forEach((name, i) => {
      const h = heights[i];
      positions[name] = { x: layerIndex * xSpacing, y: cursor + h / 2 };
      cursor += h + ySpacing;
    });
  });
  return positions;
}
