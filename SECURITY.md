# Security Policy

## Supported versions

OptunaSharp is pre-1.0 and under active development. Security fixes are applied to the latest
released version on NuGet and to `master`.

| Version | Supported |
|---------|-----------|
| latest  | ✅        |
| older   | ❌        |

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public issue.

- Preferred: open a [GitHub security advisory](https://github.com/SciSharp/OptunaSharp/security/advisories/new)
  for this repository (Security → Report a vulnerability).
- Alternatively, contact a maintainer directly.

Include a description, reproduction steps, affected version(s), and impact. We aim to acknowledge
reports within a few business days and will coordinate a fix and disclosure timeline with you.

## Scope

OptunaSharp is a .NET library. Note that:

- It depends on **NumSharp** and (for interop tests only) **pythonnet**; vulnerabilities in those
  dependencies should be reported to their respective projects.
- The `OptunaSharp.Tests.Interop` project can execute an embedded CPython + Optuna. Only run it
  against trusted Python environments.
