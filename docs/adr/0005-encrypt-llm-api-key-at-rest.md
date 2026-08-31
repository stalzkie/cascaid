---
status: accepted
---

# Encrypt the bring-your-own LLM API key at rest instead of moving it out of the DB or documenting the gap

PRD 7's opt-in LLM-generated risk explanations let a customer configure their own
LLM endpoint's API key (`cascaid.explain.configure`), which `set_config` stores as
plaintext in `Config.value` -- a real customer secret sitting unencrypted in a
key/value table any DB-level reader (a customer's own analytics team running ad hoc
queries, a misconfigured read replica, a backup snapshot) can read in full.

We're encrypting it at rest (application-level, e.g. Fernet, keyed from an env var
Cascaid itself never logs or persists) rather than the other two options: moving it out
of the DB into a required env var removes the plaintext-in-DB problem but loses the
current CLI-configure-once/persists-across-restarts convenience -- every self-hosted
customer would need to wire it into their own process/container env instead of running
one command, worse UX for the same userbase ADR 0004 optimizes for. Documenting it as a
known limitation is cheapest but leaves a real secret exposed by default, which doesn't
meet this project's accuracy/effectiveness bar for something this concrete and already
identified, not hypothetical.

## Considered Options

- **Move out of the DB, require an env var** (rejected): removes the risk entirely, but
  trades away the configure-once convenience the self-hosted userbase gets today.
- **Document as a known limitation** (rejected): zero build cost, but leaves a
  confirmed real secret in plaintext.

## Consequences

- The encryption key itself needs somewhere to live that isn't the database -- an env
  var Cascaid reads at process start, same category of "customer must provision this
  outside the DB" as the rejected option above, just for a symmetric key instead of the
  secret itself. Losing that key means losing the ability to decrypt the stored API key
  (not the data itself, since it's reconfigurable), which needs documenting.
- Only `llm_api_key` is in scope now; other `Config` values aren't secrets today, but
  any future sensitive `Config` entry should default to going through the same
  encryption helper rather than plaintext by default.
