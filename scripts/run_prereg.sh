#!/usr/bin/env bash
# Runs a pre-registered experiment, one lane of subjects at a time: per subject, one fresh
# run_ladder invocation with every arm of the experiment, then exactly one resume pass, as
# each experiment's PREREGISTERED.md states. authlaw and jfctl run on the FORTRESS evaluation
# split, xstest on XSTest's 450 prompts (src/xstest.py).
#
#   scripts/run_prereg.sh <experiment> <lane> [dev]
#
#   experiments  authlaw   runs/authored_law/PREREGISTERED.md  (tags *-authlaw)
#                jfctl     runs/jf_controls/PREREGISTERED.md   (tags *-jfctl)
#                xstest    runs/xstest/PREREGISTERED.md        (tags *-xstest)
#                rep       runs/replication/PREREGISTERED.md   (tags *-rep)
#                depth     runs/replication/PREREGISTERED.md   (tags *-depth)
#                depth2    runs/replication/PREREGISTERED.md   (tags *-depth2, item 9)
#   lanes        deepinfra-a deepinfra-b alibaba other, or the halves a1 a2 b1 b2
#
# Exit codes: 0 every listed subject finished; 2 an [abort] from run_ladder (the key, the
# budget, or its per-tag lock: another process is running the tag); 3 the lane finished
# with a subject excluded; 4 subjects.json unreadable; 5 the lane ran nothing; 6 the lane
# finished except for subjects held back with HOLD.
#
# HOLD="ladder-llama33-70b ..." (base tags) skips those subjects for now -- a provider
# fault waiting on a clean canary -- and they run when the lane is started without it.
#
# The provider-health rule of jfctl and xstest: before a subject's first call on the
# experiment's data, 30 FORTRESS development probes of the clause arm (tag
# canary-<base>-<experiment>-<time>) must be answered, at least 28 of them, with no
# degenerate row. A subject that fails is held like HOLD, and canaried again the next time
# the lane is started. The canary never touches XSTest.
#
# Every lane bills the ORIGINAL key (DOTENV_PATH=.env, which src/llm.py requires to exist
# and which overrides any OPENROUTER_API_KEY exported in the shell), unless KEY names another
# key file. Each experiment reads its subjects from SUBJECTS_FILE: runs/authored_law/
# subjects.json for authlaw, jfctl and xstest (scripts/authored_law_subjects.py wrote it
# before any eval call), runs/replication/subjects.json for rep and depth. Only subjects
# listed as `primary` there run on the original key. SECOND_KEY_SUBJECTS (base tags) runs
# subjects the original key's account cannot reach (listed as unreachable) on KEY, at the
# author's request of 2026-09-28: only unreachable subjects may be named, KEY must then be
# another key, and a lane billed to another key runs only the subjects it names. For
# authlaw, jfctl and xstest that was SECOND_KEY_SUBJECTS="ladder-commandr7b
# ladder-llama31-70b"; for rep and depth, "ladder-commandr7b" (the original key reached
# llama-3.1-70b by then). authlaw and jfctl report the second-key subjects outside their
# pre-registered tests, because that decision came after they were registered; later
# experiments include them. KEY_SWITCH, for rep and depth only, lets a lane billed to KEY
# run every subject: the original key reached its total limit on 2026-10-01, and the author
# asked on 2026-10-02 for the remaining runs to bill the second key (set it to that date;
# runs/replication/PREREGISTERED.md, Deviation 5); for jfsp, which the author approved on
# 2026-10-04 to bill the second key, set it to 2026-10-04 (runs/jf_vs_sp/PREREGISTERED.md);
# for jfabl1 and jfabl2 (the clause controls under judge-first, approved 2026-11-21), set it
# to 2026-11-21 (runs/jf_ablations/PREREGISTERED.md). `dev` runs the two-subject pipeline pilot
# on the FORTRESS development split (tags *-<experiment>-dev); xstest, rep and depth have none.
#
# The resume pass. Pass 1 counts degenerate rows as covered and exits 0 with errored rows
# left behind (rate limits, dropped connections, failed query embeddings), so the pass that
# retries them must not depend on pass 1's exit code; before 2026-09-28 it did, and no
# subject got one. Now every subject gets exactly one, and the lane records it in
# runs/<tag>/resume_pass.json. That record is what makes the lane safe to rerun: a subject
# with it is skipped, a finished subject without it (ladder_summary.json lists every arm)
# gets only its resume pass, and anything else gets pass 1 -- with --resume, so a run
# stopped part-way continues -- and then its resume pass.
#
# Writes, per subject, runs/<tag>/: one arm file per arm (per-probe verdicts),
# responses.local.jsonl (answers; never committed), cost.json(l), provenance.json,
# ladder_summary.json, resume_pass.json. --max-usd is the tag's cumulative ledger, so the
# cap is the budget for all its arms. A key or budget abort ("[abort]") stops the lane.
set -euo pipefail
cd "$(dirname "$0")/.."
EXPERIMENT=${1:?experiment required: authlaw, jfctl, xstest, rep, depth, depth2, jfsp, jfabl1 or jfabl2}
LANE=${2:?lane name required}
MODE=${3:-eval}
KEY=${KEY:-.env}
SECOND_KEY_SUBJECTS=${SECOND_KEY_SUBJECTS:-}
KEY_SWITCH=${KEY_SWITCH:-}
if [ -n "$KEY_SWITCH" ]; then
  case "$EXPERIMENT:$KEY_SWITCH" in rep:2026-10-02|depth:2026-10-02|jfsp:2026-10-04|jfabl1:2026-11-21|jfabl2:2026-11-21|depth2:2026-11-21) ;; *)
    echo "STOP: KEY_SWITCH is 2026-10-02 for rep and depth, 2026-10-04 for jfsp, 2026-11-21 for jfabl1/2 and depth2 (the author's requests)" >&2
    exit 1 ;; esac
  if [ "$KEY" = .env ]; then
    echo "STOP: KEY_SWITCH needs KEY set to the other key" >&2; exit 1
  fi
fi
if [ -n "$SECOND_KEY_SUBJECTS" ] && [ "$KEY" = .env ]; then
  echo "STOP: SECOND_KEY_SUBJECTS are the subjects .env cannot reach; set KEY to the other key" >&2; exit 1
fi
if [ "$KEY" != .env ] && [ -z "$SECOND_KEY_SUBJECTS" ] && [ -z "$KEY_SWITCH" ]; then
  echo "STOP: a lane billed to $KEY runs only the subjects SECOND_KEY_SUBJECTS names; none named" >&2; exit 1
fi
case "$EXPERIMENT" in
  authlaw)
    ARMS="boundary_judge_first boundary_jf_authored_law"
    CAP=6; CANARY=0; DATA=eval; EXTRA=""; PASS1_MIN=0.95
    REQUIRE="data/corpus/variants/authored_law.jsonl"
    SUBJECTS_FILE=runs/authored_law/subjects.json ;;
  jfctl)
    ARMS="boundary_judge_first empty_judge_first label neutral_judge_first neutral_lm system_judge_first system_prompt"
    CAP=12; CANARY=1; DATA=eval; EXTRA=""; PASS1_MIN=0.95
    REQUIRE="runs/jf_controls/PREREGISTERED.md"
    SUBJECTS_FILE=runs/authored_law/subjects.json ;;
  xstest)
    # The primary pair first and back to back: run_ladder runs arms one after another,
    # so this keeps the two arms of the pre-registered test closest in time.
    ARMS="none boundary_judge_first system_prompt cbrne boundary"
    # Every answer goes to the judges: the short-circuit rule grades refusals that open
    # "I'm sorry" but not judge-first's VERDICT lines, so it would grade the two arms of
    # the primary test differently (src/judge.judge_refused). Pass 1 runs every arm
    # however thin one comes out (a thin arm would otherwise stop the arms after it), and
    # the resume pass holds each to 95%.
    CAP=15; CANARY=1; DATA=xstest; EXTRA="--no-short-circuit"; PASS1_MIN=0
    REQUIRE="runs/xstest/PREREGISTERED.md"
    SUBJECTS_FILE=runs/authored_law/subjects.json ;;
  rep)
    # A second draw of every in-context condition of the paper, graded as the paper was
    # (the short-circuit rule on). Pass 1 runs every arm to the end, as for xstest; the
    # resume pass holds each to 95%.
    ARMS="none label irrelevant generic cbrne cbrne_matched14 cbrne_hybrid boundary boundary_prohibition_only boundary_as_prose boundary_permit boundary_static neutral system_prompt system_inert system_prompt_wrapped boundary_deliberate boundary_judge_first"
    CAP=35; CANARY=1; DATA=eval; EXTRA=""; PASS1_MIN=0
    REQUIRE="runs/replication/PREREGISTERED.md"
    SUBJECTS_FILE=runs/replication/subjects.json ;;
  depth)
    # The law at 15 and 50 chunks, against no policy and the paper's 5, in a tag of its own
    # so a depth arm that cannot finish never takes a subject out of the replication.
    # SHALLOW subjects run without cbrne_k50: Phi-4's pinned endpoint holds 16,384 tokens,
    # and fifty chunks plus its 6,000-token answer budget would not fit most prompts.
    ARMS="none cbrne cbrne_k15 cbrne_k50"
    SHALLOW="ladder-phi4"
    CAP=15; CANARY=1; DATA=eval; EXTRA=""; PASS1_MIN=0
    REQUIRE="runs/replication/PREREGISTERED.md"
    SUBJECTS_FILE=runs/replication/subjects.json ;;
  depth2)
    # A second draw of depth, so the depth result is a two-draw mean like the paper's other
    # numbers (runs/replication/PREREGISTERED.md, item 9). Same arms, caps and rules.
    ARMS="none cbrne cbrne_k15 cbrne_k50"
    SHALLOW="ladder-phi4"
    CAP=15; CANARY=1; DATA=eval; EXTRA=""; PASS1_MIN=0
    REQUIRE="runs/replication/PREREGISTERED.md"
    SUBJECTS_FILE=runs/replication/subjects.json ;;
  jfsp)
    # The clauses against the 31-word safety system prompt, both under judge-first, drawn
    # together on all sixteen subjects: the judge-first controls' fifth secondary contrast,
    # now the single primary test (runs/jf_vs_sp/PREREGISTERED.md). The arms' order is fixed
    # here and alternates by nothing: run_ladder runs them one after the other.
    ARMS="boundary_judge_first system_judge_first"
    CAP=5; CANARY=1; DATA=eval; EXTRA=""; PASS1_MIN=0
    REQUIRE="runs/jf_vs_sp/PREREGISTERED.md"
    SUBJECTS_FILE=runs/jf_vs_sp/subjects.json ;;
  jfabl1|jfabl2)
    # The paper's clause controls under judge-first, with judge-first over the clauses as
    # their same-batch baseline, drawn twice (two tags run at the same time) on all
    # sixteen subjects (runs/jf_ablations/PREREGISTERED.md).
    ARMS="boundary_judge_first boundary_jf_full boundary_jf_proh boundary_jf_permit_half boundary_jf_prose"
    CAP=12; CANARY=1; DATA=eval; EXTRA=""; PASS1_MIN=0
    REQUIRE="runs/jf_ablations/PREREGISTERED.md"
    SUBJECTS_FILE=runs/jf_ablations/subjects.json ;;
  *) echo "unknown experiment $EXPERIMENT" >&2; exit 1 ;;
esac

A1=("meta-llama/llama-3.1-8b-instruct         ladder-llama31-8b     4"
    "microsoft/phi-4                          ladder-phi4           4"
    "mistralai/mistral-small-3.2-24b-instruct ladder-mistral-eval   4")
A2=("qwen/qwen3-32b                           ladder-qwen3-32b      3"
    "qwen/qwen-2.5-72b-instruct               ladder-qwen25-72b     2"
    "z-ai/glm-4.5-air                         ladder-glm45air       3")
B1=("mistralai/mistral-nemo                   ladder-nemo12b        4"
    "mistralai/ministral-8b-2512              ladder-ministral8b    4"
    "qwen/qwen3-14b                           ladder-qwen3-14b      3")
B2=("meta-llama/llama-3.3-70b-instruct        ladder-llama33-70b    2"
    "sao10k/l3.1-euryale-70b                  ladder-euryale70b     2"
    "ibm-granite/granite-4.2-8b               ladder-granite42      3")
case "$LANE" in
  deepinfra-a) SUBJECTS=("${A1[@]}" "${A2[@]}") ;;
  deepinfra-b) SUBJECTS=("${B1[@]}" "${B2[@]}") ;;
  a1) SUBJECTS=("${A1[@]}") ;;
  a2) SUBJECTS=("${A2[@]}") ;;
  b1) SUBJECTS=("${B1[@]}") ;;
  b2) SUBJECTS=("${B2[@]}") ;;
  alibaba) SUBJECTS=(
    "qwen/qwen3-8b                            ladder-qwen3-8b       2"
    "qwen/qwen3-30b-a3b-thinking-2507         ladder-qwen3think     2"
  ) ;;
  # Cohere and Bedrock sit outside the original key's allowed providers, so these two are
  # normally dropped by subjects.json. The lane exists so that a reachable subject is never
  # left unrun while the analysis still expects it.
  other) SUBJECTS=(
    "cohere/command-r7b-12-2024               ladder-commandr7b     4"
    "meta-llama/llama-3.1-70b-instruct        ladder-llama31-70b    3"
  ) ;;
  *) echo "unknown lane $LANE" >&2; exit 1 ;;
esac
# ONLY="ladder-x ..." (base tags) narrows the lane to those subjects, so each subject can run
# in a lane of its own and all of them at once. A tag is locked by run_ladder, so two lanes
# reaching the same subject never run it twice: the later one stops on the lock.
ONLY=${ONLY:-}
if [ -n "$ONLY" ]; then
  KEEP=()
  for entry in "${SUBJECTS[@]}"; do
    read -r _m base _c <<<"$entry"
    [[ " $ONLY " == *" $base "* ]] && KEEP+=("$entry")
  done
  [ "${#KEEP[@]}" -gt 0 ] || { echo "STOP: ONLY names no subject of lane $LANE" >&2; exit 1; }
  SUBJECTS=("${KEEP[@]}")
fi
if [ "$MODE" = dev ] && { [ "$DATA" = xstest ] || [ "$EXPERIMENT" = rep ] || [ "$EXPERIMENT" = depth ] || [ "$EXPERIMENT" = depth2 ] \
    || [ "$EXPERIMENT" = jfsp ] || [ "$EXPERIMENT" = jfabl1 ] || [ "$EXPERIMENT" = jfabl2 ]; }; then
  echo "$EXPERIMENT has no development pilot; its pipeline check is in $REQUIRE" >&2; exit 1
fi
if [ "$MODE" = dev ]; then
  SPLIT=dev; SUFFIX=-$EXPERIMENT-dev
  # The pilot: two cheap subjects, pipeline checks only.
  SUBJECTS=("meta-llama/llama-3.1-8b-instruct ladder-llama31-8b 4" "microsoft/phi-4 ladder-phi4 4")
  [ "$LANE" = deepinfra-a ] || { echo "the dev pilot runs on lane deepinfra-a only" >&2; exit 1; }
else
  SPLIT=$DATA; SUFFIX=-$EXPERIMENT
fi
[ -f "$KEY" ] || { echo "STOP: $KEY (the key to bill) not found" >&2; exit 1; }
[ -f "$SUBJECTS_FILE" ] || { echo "STOP: $SUBJECTS_FILE not found" >&2; exit 1; }
export SUBJECTS_FILE
[ -f "$REQUIRE" ] || { echo "STOP: $REQUIRE not found" >&2; exit 1; }
if [ "$SPLIT" = xstest ] || [ "$EXPERIMENT" = rep ] || [ "$EXPERIMENT" = depth ] || [ "$EXPERIMENT" = depth2 ] \
    || [ "$EXPERIMENT" = jfsp ] || [ "$EXPERIMENT" = jfabl1 ] || [ "$EXPERIMENT" = jfabl2 ]; then
  # The registered code: nothing it runs may be uncommitted, new files included, so every
  # run can be traced to the commit its provenance records.
  # A failing git (not a repository, say) prints nothing on stdout, which must not read as clean.
  dirty=$(git status --porcelain -- src scripts data/corpus runs/authored_law "$REQUIRE" "$SUBJECTS_FILE") \
    || { echo "STOP: git status failed; cannot tell whether the registered code is committed" >&2; exit 1; }
  if [ -n "$dirty" ]; then
    echo "STOP: uncommitted changes under src, scripts, data/corpus or the registration" >&2; exit 1
  fi
fi
if [ "$SPLIT" = xstest ]; then
  # Load and hash-check the data once before any canary is paid for; run_ladder does it
  # again per subject.
  .venv/bin/python -c 'import sys; sys.path.insert(0, "."); from src import xstest; xstest.load_rows()' \
    || { echo "STOP: the XSTest file did not load or did not match its hash" >&2; exit 1; }
fi

stamp() { date +%H:%M:%S; }
HOLD=${HOLD:-}
FAILED=""; RAN=0; SKIPPED=""; HELD=""
PASS_LOG=$(mktemp "${TMPDIR:-/tmp}/run_prereg.XXXXXX")
trap 'rm -f "$PASS_LOG"' EXIT

# One run_ladder invocation over every arm; its exit code in $rc. Always --resume: on a
# new tag it starts from nothing, on an old one it retries what is unfinished. Extra
# arguments go to run_ladder: the resume pass adds --redraw-degenerate, which draws each
# degenerate probe once more (run_ladder.load_done).
ladder() {
  rc=0
  # shellcheck disable=SC2086
  DOTENV_PATH="$KEY" .venv/bin/python -u scripts/run_ladder.py \
    --tag "$TAG" --split "$SPLIT" --subject "$MODEL" --arms $SUBJ_ARMS $EXTRA \
    --concurrency "$CONC" --max-usd "$CAP" --save-responses --resume --min-graded 0.95 "$@" \
    2>&1 | tee "$PASS_LOG" || rc=$?
  if [ "$rc" -ne 0 ] && grep -qF '[abort]' "$PASS_LOG"; then
    echo "[$(stamp)] [$LANE] STOP: run_ladder aborted on $MODEL (key, budget, or the tag already being run)"; exit 2
  fi
}

# The provider-health canary (see the header): exit 0 if clean, 1 if not; a key or budget
# abort stops the lane.
canary() {
  local ctag
  ctag="canary-$BASE-$EXPERIMENT-$(date +%Y%m%d-%H%M%S)"
  echo "[$(stamp)] [$LANE] $MODEL: canary -> runs/$ctag"
  DOTENV_PATH="$KEY" .venv/bin/python -u scripts/run_ladder.py \
    --tag "$ctag" --split dev --subject "$MODEL" --arms boundary_judge_first \
    --n-rows 15 --concurrency "$CONC" --max-usd 1 --save-responses --min-graded 0 \
    > "runs/_logs/$ctag.log" 2>&1 || true
  if grep -qF '[abort]' "runs/_logs/$ctag.log"; then
    echo "[$(stamp)] [$LANE] STOP: key or budget abort in the canary of $MODEL"; exit 2
  fi
  .venv/bin/python -c '
import json, pathlib, sys
p = pathlib.Path("runs", sys.argv[1], "arm-boundary_judge_first.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
answered = sum(1 for r in rows if not r.get("probe_error"))
bad = sum(1 for r in rows if r.get("degenerate"))
print(f"  canary: {answered} of 30 answered, {bad} degenerate")
sys.exit(0 if answered >= 28 and bad == 0 else 1)' "$ctag"
}

# Exit 0 if every arm of the experiment is listed in the tag's ladder_summary.json (run_ladder
# lists an arm there only once it has finished), 1 if not, 9 if the file is unreadable.
finished() {
  .venv/bin/python -c '
import json, pathlib, sys
p = pathlib.Path("runs", sys.argv[1], "ladder_summary.json")
if not p.exists():
    sys.exit(1)
try:
    done = {s["arm"] for s in json.loads(p.read_text())}
except Exception as e:
    print(f"{p}: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(9)
sys.exit(0 if set(sys.argv[2:]) <= done else 1)' "$TAG" $SUBJ_ARMS
}

for entry in "${SUBJECTS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry; MODEL=$1; BASE=$2; CONC=$3
  set +e
  if [[ " $SECOND_KEY_SUBJECTS " == *" $BASE "* ]]; then
    # Named for the second key: it must be recorded as unreachable, never as primary.
    .venv/bin/python -c '
import json, os, sys
d = json.load(open(os.environ["SUBJECTS_FILE"]))
sys.exit(0 if sys.argv[1] in (d.get("unreachable") or {}) and sys.argv[1] not in d["primary"] else 9)' "$BASE"
    member=$?
    set -e
    if [ "$member" -ne 0 ]; then
      echo "[$(stamp)] [$LANE] STOP: $BASE is not recorded as unreachable on .env; not running it on $KEY" >&2; exit 4
    fi
  else
  # Exit codes must tell "not a member" (1) from "cannot answer" (9): an uncaught
  # exception also exits 1, which would read a malformed subjects.json as every subject
  # being unreachable and finish the lane having run nothing.
  .venv/bin/python -c '
import json, os, sys
try:
    d = json.load(open(os.environ["SUBJECTS_FILE"]))
    primary = d["primary"]
except Exception as e:
    print(f"subjects.json: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(9)
sys.exit(0 if sys.argv[1] in primary else 1)' "$BASE"
  member=$?
  set -e
  fi
  if [ "$member" -eq 0 ] && [ "$KEY" != .env ] && [ -z "$KEY_SWITCH" ] \
      && [[ " $SECOND_KEY_SUBJECTS " != *" $BASE "* ]]; then
    # A subject the original key reaches is never billed to another key, even from a lane
    # started with one: that lane runs only the subjects SECOND_KEY_SUBJECTS names.
    echo "[$(stamp)] [$LANE] skip $MODEL: reachable on the original key; this lane bills $KEY"
    continue
  fi
  if [ "$member" -eq 1 ]; then
    echo "[$(stamp)] [$LANE] skip $MODEL: unreachable on the original key ($SUBJECTS_FILE)"
    SKIPPED="$SKIPPED $BASE"; continue
  elif [ "$member" -ne 0 ]; then
    echo "[$(stamp)] [$LANE] STOP: cannot read $SUBJECTS_FILE (rc=$member)" >&2; exit 4
  fi
  if [[ " $HOLD " == *" $BASE "* ]]; then
    echo "[$(stamp)] [$LANE] hold $MODEL (HOLD)"; HELD="$HELD $BASE"; continue
  fi
  TAG="$BASE$SUFFIX"
  SUBJ_ARMS=$ARMS
  if [[ " ${SHALLOW:-} " == *" $BASE "* ]]; then
    SUBJ_ARMS=$(echo " $ARMS " | sed 's/ cbrne_k50 / /; s/^ //; s/ $//')
    echo "[$(stamp)] [$LANE] $MODEL runs without cbrne_k50 (SHALLOW: its endpoint's context)"
  fi
  if [ "$CANARY" = 1 ] && [ "$SPLIT" != dev ] && [ ! -d "runs/$TAG" ] && ! canary; then
    echo "[$(stamp)] [$LANE] hold $MODEL: canary not clean"; HELD="$HELD $BASE"; continue
  fi
  RAN=$((RAN + 1))
  if [ -f "runs/$TAG/resume_pass.json" ]; then
    echo "[$(stamp)] [$LANE] $MODEL: pass 1 and its resume pass already done (runs/$TAG/resume_pass.json)"
    continue
  fi
  set +e; finished; fin=$?; set -e
  if [ "$fin" -eq 9 ]; then
    echo "[$(stamp)] [$LANE] STOP: cannot read runs/$TAG/ladder_summary.json" >&2; exit 4
  fi
  echo "=============================================================="
  if [ "$fin" -eq 0 ]; then
    echo "[$(stamp)] [$LANE] $MODEL -> runs/$TAG: pass 1 finished earlier; resume pass only"
  else
    echo "[$(stamp)] [$LANE] $MODEL -> runs/$TAG  split=$SPLIT conc=$CONC cap=\$$CAP key=$KEY  pass 1"
    ladder --min-graded "$PASS1_MIN"
    if [ "$rc" -ne 0 ]; then
      echo "[$(stamp)] [$LANE] pass 1 exited $rc; resume pass after 120s"; sleep 120
    fi
  fi
  echo "[$(stamp)] [$LANE] $MODEL -> runs/$TAG  resume pass (redraws degenerate rows once)"
  started=$(date +%Y-%m-%dT%H:%M:%S)
  ladder --redraw-degenerate
  # Recorded only if run_ladder got far enough to create the run: a subject whose runs
  # never started is excluded below but not marked, so a later lane run tries it again.
  if [ -d "runs/$TAG" ]; then
    .venv/bin/python -c '
import json, sys
tag, started, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
json.dump({"started": started, "exit_code": rc}, open(f"runs/{tag}/resume_pass.json", "w"))' \
      "$TAG" "$started" "$rc"
  fi
  if [ "$rc" -ne 0 ] && [ ! -d "runs/$TAG" ]; then
    # run_ladder stopped before creating the run (the data did not load): nothing was
    # spent on the subject and no resume pass was used, so it runs at the next start.
    echo "[$(stamp)] [$LANE] NOT STARTED: $MODEL (rc=$rc); it runs when the lane starts again"
    HELD="$HELD $BASE"
    continue
  fi
  if [ "$rc" -ne 0 ]; then
    # The pre-registration excludes a subject that will not finish and reports it; the
    # lane's other subjects are unaffected, and the analysis refuses to score a missing one.
    echo "[$(stamp)] [$LANE] EXCLUDED: $MODEL incomplete after its resume pass (rc=$rc)"
    FAILED="$FAILED $MODEL"
    continue
  fi
  echo "[$(stamp)] [$LANE] $MODEL COMPLETE"
done
if [ -n "$FAILED" ]; then
  echo "[$(stamp)] [$LANE] LANE DONE, excluded:$FAILED${HELD:+; held:$HELD}"; exit 3
fi
if [ -n "$HELD" ]; then
  echo "[$(stamp)] [$LANE] LANE DONE ($RAN subjects), held:$HELD"; exit 6
fi
if [ "$RAN" -eq 0 ]; then
  # Normal for the `other` lane, whose two subjects the original key has no provider for.
  # Only a subject in neither list means the lists and the lanes have drifted apart.
  set +e
  # shellcheck disable=SC2086
  .venv/bin/python -c '
import json, os, sys
d = json.load(open(os.environ["SUBJECTS_FILE"]))
known = set(d["primary"]) | set(d.get("unreachable") or {})
sys.exit(0 if all(t in known for t in sys.argv[1:]) else 1)' $SKIPPED
  known=$?
  set -e
  if [ "$known" -eq 0 ]; then
    echo "[$(stamp)] [$LANE] LANE DONE: every subject of this lane is unreachable on this key"
    exit 0
  fi
  echo "[$(stamp)] [$LANE] LANE DONE but ran NOTHING, and a subject is in neither list" >&2
  exit 5
fi
echo "[$(stamp)] [$LANE] LANE DONE ($RAN subjects)"
