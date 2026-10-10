#!/usr/bin/env bash
# redate.sh — restamp feature-branch commits across a window of working days.
#
# Run from repo root in WSL/Git Bash before opening a PR:
#   bash scripts/redate.sh [branch] [base]
#
#   branch  feature branch to redate  (default: current branch)
#   base    merge target               (default: main)
#
# Environment:
#   PR_DATE      last day of the window (default: today)
#   SPREAD_DAYS  how many days back the window reaches (default: 1 = PR_DATE only)
#   DAY_START    first commit hour each day (default: 09)
#   DAY_END      last commit hour each day  (default: 17)
#
# Examples:
#   bash scripts/redate.sh                      # all commits on today
#   SPREAD_DAYS=28 bash scripts/redate.sh       # spread across the last 4 weeks
#
# A window longer than one day makes the history say the work happened on days it did
# not. That is the owner's call to make, not this script's, but it should be a deliberate
# flag rather than the default — so SPREAD_DAYS defaults to 1 and the summary below prints
# the window it is about to write.

set -euo pipefail

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
BASE="${2:-main}"
PR_DATE="${PR_DATE:-$(date +%Y-%m-%d)}"
SPREAD_DAYS="${SPREAD_DAYS:-1}"
DAY_START="${DAY_START:-09}"
DAY_END="${DAY_END:-17}"

if [[ ! "$SPREAD_DAYS" =~ ^[0-9]+$ ]] || [[ "$SPREAD_DAYS" -lt 1 ]]; then
  echo "SPREAD_DAYS must be a positive integer, got '$SPREAD_DAYS'" >&2
  exit 1
fi

# Collect commits oldest → newest
mapfile -t COMMITS < <(git log --reverse "${BASE}..${BRANCH}" --format="%H")
N=${#COMMITS[@]}

if [[ $N -eq 0 ]]; then
  echo "No commits ahead of $BASE on $BRANCH — nothing to redate."
  exit 0
fi

FIRST_DAY=$(date -d "$PR_DATE -$((SPREAD_DAYS - 1)) days" +%Y-%m-%d)

echo "Redating $N commits on '$BRANCH'"
echo "  window:  $FIRST_DAY .. $PR_DATE  ($SPREAD_DAYS day(s))"
echo "  hours:   ${DAY_START}:00-${DAY_END}:00 each day"

# Map each commit to a day, oldest first, so the log still reads in the order the work
# was done. The index is scaled across the whole window rather than filled a day at a
# time: dealing ceil(N/days) per day leaves the tail of the window empty, which would put
# the newest commit several days before the PR that contains it.
declare -a DAY_OF SLOT_OF COUNT_ON SEEN
for i in "${!COMMITS[@]}"; do
  d=$(( i * SPREAD_DAYS / N ))
  (( d >= SPREAD_DAYS )) && d=$(( SPREAD_DAYS - 1 ))
  DAY_OF[$i]=$d
  COUNT_ON[$d]=$(( ${COUNT_ON[$d]:-0} + 1 ))
done

# Position within its own day, so each day's commits are spread over the working hours
# rather than bunched at the start.
for i in "${!COMMITS[@]}"; do
  d=${DAY_OF[$i]}
  SLOT_OF[$i]=$(( ${SEEN[$d]:-0} ))
  SEEN[$d]=$(( ${SEEN[$d]:-0} + 1 ))
done

USED_DAYS=0
for d in $(seq 0 $(( SPREAD_DAYS - 1 ))); do
  [[ -n "${COUNT_ON[$d]:-}" ]] && USED_DAYS=$(( USED_DAYS + 1 ))
done
echo "  days used: $USED_DAYS of $SPREAD_DAYS"
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

  DAY_INDEX=${DAY_OF[$i]}
  SLOT=${SLOT_OF[$i]}
  ON_THIS_DAY=${COUNT_ON[$DAY_INDEX]}
  DAY=$(date -d "$FIRST_DAY +${DAY_INDEX} days" +%Y-%m-%d)

  DAY_START_TS=$(date -d "$DAY ${DAY_START}:00:00" +%s)
  DAY_END_TS=$(date -d "$DAY ${DAY_END}:00:00" +%s)

  if [[ $ON_THIS_DAY -le 1 ]]; then
    DAY_STEP=0
  else
    DAY_STEP=$(( (DAY_END_TS - DAY_START_TS) / (ON_THIS_DAY - 1) ))
  fi

  TS=$(( DAY_START_TS + SLOT * DAY_STEP ))
  D="$(date -d "@${TS}" '+%Y-%m-%dT%H:%M:%S %z')"

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
