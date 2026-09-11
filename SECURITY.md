# Security Policy

Cascaid is self-hosted: it runs inside your own environment against your own
pipeline data, and nothing is sent to a third party by design. That doesn't
mean it's risk-free -- it terminates a dashboard and API over HTTP, persists
to Postgres, and exposes an MCP server, so vulnerabilities in auth, ingestion,
or the API surface are still worth taking seriously.

## Supported Versions

Cascaid is pre-1.0 (`0.x`, alpha). Security fixes are made against the latest
release on `master`; there is no separate maintenance branch yet.

| Version | Supported |
| ------- | --------- |
| latest `0.x` | ✅ |
| anything older | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately using one of:

- [GitHub Security Advisories](https://github.com/stalzkie/cascaid/security/advisories/new)
  for this repo (preferred -- keeps the report and any discussion private
  until a fix ships).
- Email dstalingrad@gmail.com with details and, if possible, steps to
  reproduce.

Please include:

- The affected component (ingestion, serving API, dashboard, MCP server,
  auth, training/models) and version or commit.
- Steps to reproduce, or a proof of concept.
- The impact you believe it has (e.g. auth bypass, data exposure, RCE).

## What to Expect

- Acknowledgement within a few days.
- We'll work with you to confirm the issue, assess severity, and prepare a
  fix before any public disclosure.
- Credit in the release notes / advisory, if you'd like it.

## Scope

In scope: the `cascaid` package, the FastAPI serving/dashboard apps, the MCP
server, the auth layer, the Docker Compose deployment, and the training
pipeline as shipped in this repository.

Out of scope: vulnerabilities in third-party dependencies (report those
upstream) and issues that require an attacker to already have direct database
or host access in a deployment you control.
