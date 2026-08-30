## Decision priorities

Every design, scope, and build-vs-integrate decision in this project is judged against four criteria: **effectiveness** (does it solve the real problem), **efficiency** (lowest cost to build and maintain), **accuracy** (correct, verifiable behavior), and **UX** (simple and pleasant for whoever consumes it — end user or integrator). When these trade off against each other, prefer the option that scores well across all four over one that maximizes a single criterion at the others' expense — e.g. don't build the more "complete" or "standard" solution if it costs materially more effort for the same effectiveness and UX.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
