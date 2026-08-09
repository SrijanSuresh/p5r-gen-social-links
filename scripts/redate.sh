#!/usr/bin/env bash
# redate.sh — spread feature-branch commits evenly across a 2-week window.
#
# Run from repo root in WSL/Git Bash before opening a PR:
#   bash scripts/redate.sh [branch] [base]
#
#   branch  feature branch to redate  (default: current branch)
#   base    merge target               (default: main)
#
# Set PR_DATE env var to override end date (default: today).
# Example: PR_DATE=2026-08-08 bash scripts/redate.sh

set -euo pipefail

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
BASE="${2:-main}"
PR_DATE="${PR_DATE:-$(date +%Y-%m-%d)}"

# Collect commits oldest → newest
mapfile -t COMMITS < <(git log --reverse "${BASE}..${BRANCH}" --format="%H")
N=${#COMMITS[@]}

if [[ $N -eq 0 ]]; then
  echo "No commits ahead of $BASE on $BRANCH — nothing to redate."
  exit 0
fi

echo "Redating $N commits on '$BRANCH' → window: $PR_DATE -14d to $PR_DATE"

# 14-day window ending on PR date
END_TS=$(date -d "$PR_DATE 17:00:00" +%s)
START_TS=$(date -d "$PR_DATE -14 days 09:00:00" +%s)

if [[ $N -eq 1 ]]; then
  STEP=0
else
  STEP=$(( (END_TS - START_TS) / (N - 1) ))
fi

echo "Interval between commits: ~$((STEP / 3600))h $((STEP % 3600 / 60))m"
echo ""

# Stash any uncommitted changes so cherry-pick doesn't trip on them
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  git stash push -u -m "redate-script-stash"
  STASHED=1
fi

# Create a scratch branch from base
TMP="redate-tmp-$$"
git checkout "$BASE"
git checkout -b "$TMP"

for i in "${!COMMITS[@]}"; do
  H="${COMMITS[$i]}"
  TS=$(( START_TS + i * STEP ))
  D="$(date -d "@${TS}" '+%Y-%m-%dT%H:%M:%S +0000')"

  git cherry-pick "$H"
  GIT_COMMITTER_DATE="$D" git commit --amend --no-edit --date="$D"
  echo "  [$((i+1))/$N] $D  ${H:0:7}  $(git log -1 --format='%s' HEAD)"
done

# Move the feature branch to the rewritten tip
git branch -f "$BRANCH" HEAD
git checkout "$BRANCH"
git branch -D "$TMP"

# Restore stash if we created one
if [[ $STASHED -eq 1 ]]; then
  git stash pop
fi

echo ""
echo "Redate complete. Verify:"
echo "  git log --format='%h %ai %s' ${BASE}..${BRANCH}"
echo ""
echo "Then force-push:"
echo "  git push --force origin ${BRANCH}"
