#!/usr/bin/env bash
# harvest.sh — discover battle-tested Optuna usage on GitHub to seed interop parity tests.
# Part of the `interop-test-porting` skill. gh is called with ambient tokens unset (keyring auth).
#
# Usage:
#   harvest.sh repos [QUERY] [LIMIT]   Popular repos importing optuna, ranked by stars.
#                                      Output: STARS<TAB>owner/repo<TAB>representative-path
#                                      (QUERY default 'import optuna', LIMIT default 100, max 1000)
#   harvest.sh gists [QUERY]           Public gists matching QUERY, via HTML scrape (no gist API).
#                                      Output: owner/gist_id  (QUERY default 'optuna create_study')
#   harvest.sh gists-ranked [LIMIT] [QUERY...]
#                                      Star-ranked gists (GraphQL Gist.stargazerCount — REST/HTML give
#                                      no stars) across a pool gathered from QUERY... (default: several
#                                      optuna queries). Output: STARS<TAB>owner/gist_id  (LIMIT default 25)
#   harvest.sh file  OWNER/REPO PATH   Print a raw file from a repo (default branch).
#   harvest.sh gist  GIST_ID [FILE]    Gist description + file list; or, with FILE, that file's content.
#
# Notes / limits (GitHub, not this script):
#   * code search returns NO stars (hence the two-hop enrich), is rate-limited 10/min, and caps at
#     1000 results ranked by relevance not popularity — treat the output as a sample, not a census.
#   * core REST (star enrichment) budget is 5000/hr — a full 1000-result run stays well under it.

GH() { env -u GH_TOKEN -u GITHUB_TOKEN gh "$@"; }

urlenc() { printf '%s' "$1" | sed 's/ /+/g'; }

cmd_repos() {
  local q="${1:-import optuna}" limit="${2:-100}"
  local tmp; tmp="$(mktemp -d)"
  # 1) code hits: owner/repo<TAB>path  (paths let you jump straight to the usage)
  GH search code "$q" --language python --limit "$limit" \
     --json path,repository --jq '.[] | "\(.repository.nameWithOwner)\t\(.path)"' 2>/dev/null \
     > "$tmp/hits.tsv"
  # 2) enrich each UNIQUE repo with its star count
  : > "$tmp/stars.tsv"
  while IFS= read -r repo; do
    [ -z "$repo" ] && continue
    local s; s="$(GH api "repos/$repo" --jq '.stargazers_count' 2>/dev/null)"
    printf '%s\t%s\n' "${s:-NA}" "$repo" >> "$tmp/stars.tsv"
  done < <(cut -f1 "$tmp/hits.tsv" | sort -u)
  # 3) join stars back onto the first path seen per repo (awk join — avoids grep -P locale issues), sort desc
  awk -F'\t' 'NR==FNR{star[$2]=$1; next} !seen[$1]++ {print star[$1]"\t"$1"\t"$2}' \
      "$tmp/stars.tsv" "$tmp/hits.tsv" | sort -t"$(printf '\t')" -k1,1nr
  rm -rf "$tmp"
}

cmd_gists() {
  local q="${1:-optuna create_study}"
  # gist code search has no API — scrape the HTML search page (plain curl, no JS needed).
  curl -s -A "Mozilla/5.0" "https://gist.github.com/search?q=$(urlenc "$q")" 2>/dev/null \
    | grep -oE 'href="/[A-Za-z0-9_-]+/[0-9a-f]{20,}"' \
    | sed -E 's#href="/(.*)"#\1#' | sort -u
}

cmd_file() {
  # OWNER/REPO PATH → raw bytes of the file on the default branch
  GH api "repos/$1/contents/$2" -H "Accept: application/vnd.github.raw" 2>/dev/null
}

cmd_gist() {
  # GIST_ID [FILE] → file list (+desc), or one file's content
  if [ "$#" -ge 2 ]; then
    GH api "gists/$1" --jq ".files[\"$2\"].content" 2>/dev/null
  else
    GH api "gists/$1" --jq '{desc: .description, files: (.files|keys)}' 2>/dev/null
  fi
}

cmd_gists_ranked() {
  # [LIMIT] [QUERY...] → STARS<TAB>owner/gist_id, descending. Gists carry no stars in REST/HTML, so
  # gather a pool from the HTML search (several queries x pages) then enrich each via GraphQL.
  local limit="${1:-25}"; shift 2>/dev/null || true
  local queries=("$@")
  [ "${#queries[@]}" -eq 0 ] && queries=("import optuna" "optuna study" "optuna create_study" \
    "optuna TPESampler" "optuna suggest_float" "optuna objective trial" "optuna pruner")
  local tmp; tmp="$(mktemp -d)"
  local q p
  for q in "${queries[@]}"; do
    for p in 1 2 3; do
      curl -s -A "Mozilla/5.0" "https://gist.github.com/search?q=$(urlenc "$q")&p=$p" 2>/dev/null \
        | grep -oE 'href="/[A-Za-z0-9_-]+/[0-9a-f]{20,}"' | sed -E 's#href="/(.*)"#\1#'
    done
  done | sort -u > "$tmp/ids.txt"
  local GQ='query($login:String!,$name:String!){user(login:$login){gist(name:$name){stargazerCount}}}'
  local owner id sc
  while IFS=/ read -r owner id; do
    [ -z "$id" ] && continue
    sc="$(GH api graphql -f query="$GQ" -F login="$owner" -F name="$id" \
        --jq '.data.user.gist.stargazerCount // 0' 2>/dev/null)"
    printf '%s\t%s/%s\n' "${sc:-0}" "$owner" "$id" >> "$tmp/stars.tsv"
  done < "$tmp/ids.txt"
  sort -t"$(printf '\t')" -k1,1nr "$tmp/stars.tsv" | head -n "$limit"
  rm -rf "$tmp"
}

case "${1:-help}" in
  repos)        shift; cmd_repos "$@" ;;
  gists)        shift; cmd_gists "$@" ;;
  gists-ranked) shift; cmd_gists_ranked "$@" ;;
  file)         shift; cmd_file  "$@" ;;
  gist)         shift; cmd_gist  "$@" ;;
  *)            sed -n '2,29p' "$0" ;;   # print the usage header
esac
