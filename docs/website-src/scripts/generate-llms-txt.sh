#!/usr/bin/env bash
# Generate an llms.txt (https://llmstxt.org/) index for the built DocFX site.
#
# Usage: scripts/generate-llms-txt.sh <site-output-dir>
#   e.g. from docs/website-src:  ./scripts/generate-llms-txt.sh ../website
#
# Emits <site-output-dir>/llms.txt — a compact, link-first summary an LLM can read to find the
# canonical docs. Deliberately minimal (no HTML parsing) so it cannot break the docs build.
set -euo pipefail

OUT_DIR="${1:-../website}"
SITE_BASE="https://scisharp.github.io/OptunaSharp"
SRC_DOCS_DIR="$(cd "$(dirname "$0")/../docs" && pwd)"

mkdir -p "$OUT_DIR"
LLMS="$OUT_DIR/llms.txt"

{
  echo "# OptunaSharp"
  echo ""
  echo "> A .NET port of Optuna (the define-by-run hyperparameter optimization framework), built on"
  echo "> NumSharp. Optuna is the behavioral source of truth. Status: scaffolding — no algorithms"
  echo "> ported yet; the build/test/package/CI/docs stack is in place."
  echo ""
  echo "## Docs"
  echo ""
  # List each conceptual doc page by its first H1, linked to the published site.
  for md in "$SRC_DOCS_DIR"/*.md; do
    [ -e "$md" ] || continue
    base="$(basename "${md%.md}")"
    title="$(grep -m1 '^# ' "$md" 2>/dev/null | sed 's/^# //')"
    [ -n "$title" ] || title="$base"
    echo "- [$title]($SITE_BASE/docs/$base.html)"
  done
  echo ""
  echo "## Reference"
  echo ""
  echo "- [API reference]($SITE_BASE/api/index.html)"
  echo "- [Source (GitHub)](https://github.com/SciSharp/OptunaSharp)"
  echo "- [NuGet](https://www.nuget.org/packages/OptunaSharp)"
} > "$LLMS"

echo "Wrote $LLMS"
