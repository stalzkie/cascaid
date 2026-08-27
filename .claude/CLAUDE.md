# graphify
- **graphify** (`.claude/skills/graphify/SKILL.md`) - any input to knowledge graph. Trigger: `/graphify`
When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

# mattpocock-skills (mandatory throughout this project)
The `mattpocock-skills` plugin is installed and must be used proactively across this entire project, in addition to graphify. Invoke the matching skill before doing the related work, without waiting to be asked:
- **mattpocock-skills:diagnosing-bugs** - whenever something is broken, throwing, failing, or slow, or the user says "diagnose"/"debug this".
- **mattpocock-skills:tdd** - whenever building a feature or fixing a bug; write tests first (red-green-refactor).
- **mattpocock-skills:prototype** - when sanity-checking a state model, logic, or UI design before committing to it.
- **mattpocock-skills:research** - when a topic, doc, or API needs investigating and the findings should be captured as a Markdown file in the repo.
- **mattpocock-skills:domain-modeling** - when discussing project terminology, or writing/editing CONTEXT.md or an ADR.
- **mattpocock-skills:codebase-design** - when designing or improving a module's interface, or deciding where a seam goes.
- **mattpocock-skills:code-review** - when reviewing changes since a commit/branch/tag, a PR, or work-in-progress.
- **mattpocock-skills:resolving-merge-conflicts** - whenever there is an in-progress git merge/rebase conflict.
- **mattpocock-skills:wizard** - when provisioning infrastructure, credentials, CI secrets, or walking a one-off migration/cutover that needs the user's hands.
- **mattpocock-skills:grilling** - when the user wants their plan, decision, or idea stress-tested.
- **mattpocock-skills:writing-for-agents** - when creating or editing skills, or modifying AGENTS.md/CLAUDE.md.
