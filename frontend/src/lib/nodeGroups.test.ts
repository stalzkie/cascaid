import { describe, expect, it } from "vitest";
import { groupPipelineNodes } from "./nodeGroups";
import type { PipelineNode } from "../types";

const node = (name: string, type: PipelineNode["type"]): PipelineNode => ({ name, type, risk_score: null });

describe("groupPipelineNodes", () => {
  it("absorbs an agent's leaf children into one group when it has two or more", () => {
    const nodes = [node("orchestrator", "agent"), node("search", "tool"), node("db", "vector_store")];
    const edges: [string, string][] = [
      ["orchestrator", "search"],
      ["orchestrator", "db"],
    ];

    const { groups, edges: groupEdges } = groupPipelineNodes(nodes, edges);

    expect(groups).toHaveLength(1);
    expect(groups[0].key).toBe("orchestrator");
    expect(groups[0].children.map((c) => c.name).sort()).toEqual(["db", "search"]);
    expect(groupEdges).toEqual([]);
  });

  it("leaves a single child un-absorbed -- grouping only kicks in for two or more", () => {
    const nodes = [node("agent", "agent"), node("store", "vector_store")];
    const edges: [string, string][] = [["agent", "store"]];

    const { groups, edges: groupEdges } = groupPipelineNodes(nodes, edges);

    expect(groups.map((g) => g.key).sort()).toEqual(["agent", "store"]);
    expect(groups.every((g) => g.children.length === 0)).toBe(true);
    expect(groupEdges).toEqual([["agent", "store"]]);
  });

  it("does not absorb a child that has another inbound edge (fan-in)", () => {
    const nodes = [node("a", "agent"), node("b", "agent"), node("shared", "tool"), node("only-a", "tool")];
    const edges: [string, string][] = [
      ["a", "shared"],
      ["b", "shared"],
      ["a", "only-a"],
    ];

    const { groups } = groupPipelineNodes(nodes, edges);
    const shared = groups.find((g) => g.key === "shared");
    expect(shared).toBeDefined();
    const a = groups.find((g) => g.key === "a");
    expect(a?.children.map((c) => c.name)).not.toContain("shared");
  });

  it("does not absorb a child that itself has outgoing edges", () => {
    const nodes = [node("agent", "agent"), node("mid", "tool"), node("leaf", "tool"), node("other", "model_endpoint")];
    const edges: [string, string][] = [
      ["agent", "mid"],
      ["agent", "other"],
      ["mid", "leaf"],
    ];

    const { groups } = groupPipelineNodes(nodes, edges);
    const agentGroup = groups.find((g) => g.key === "agent");
    expect(agentGroup?.children.map((c) => c.name)).not.toContain("mid");
  });

  it("never absorbs another agent, even as a lone downstream node", () => {
    const nodes = [node("a", "agent"), node("b", "agent"), node("t", "tool")];
    const edges: [string, string][] = [
      ["a", "b"],
      ["a", "t"],
    ];

    const { groups } = groupPipelineNodes(nodes, edges);
    expect(groups.map((g) => g.key).sort()).toEqual(["a", "b", "t"]);
  });

  it("keeps every node reachable through exactly one group", () => {
    const nodes = [node("solo", "tool")];
    const { groups, edges } = groupPipelineNodes(nodes, []);
    expect(groups).toEqual([{ key: "solo", parent: nodes[0], children: [] }]);
    expect(edges).toEqual([]);
  });
});
