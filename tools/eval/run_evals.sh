#!/bin/bash
# The standard measurement suite for one trained run. Results go to runs/<run>/eval/.
#
#   bash tools/eval/run_evals.sh rp_run24a          (from the repo root; any Python env
#   PY=.venv/bin/python bash tools/eval/run_evals.sh rp_run24a     with jax + mujoco)
#
# Runs one check at a time (on a GPU each would otherwise grab most of its memory).
# pod_run.sh calls this right after training, so the pod does the slow checks while
# it's still warm. The GIF needs an OpenGL context: on a pod set GIF_GL=egl (only the
# GIF step gets it: with MUJOCO_GL set, importing mujoco loads that GL library).
RUN=$1
PY=${PY:-python}
cd "$(dirname "$0")/../.."   # repo root
OUT=runs/$RUN/eval
mkdir -p $OUT
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # take GPU memory as needed, not 75% up front
F='^(WARNING|W[0-9]{4}|I[0-9]{4})|solver iterations|warn_overflow|[Ww]arning'
T=$OUT/times.txt
echo "evals start $(date -u +%H:%M:%S)" > $T

check() {  # check <name> <command...>: run it, keep the output, time it
  local name=$1; shift
  local t=$(date +%s)
  "$@" 2>&1 | grep -v -E "$F" > $OUT/$name.txt
  echo "$name: $(( $(date +%s) - t )) s" >> $T
}

check lift      $PY tools/eval/lift_check.py $RUN 0.12
check knee      $PY tools/eval/knee_margin.py $RUN
check hardware  $PY tools/eval/hardware_quick.py $RUN
check terrain   $PY tools/eval/terrain_eval.py $RUN 16 0.12 0,3,6,9
check stress    $PY tools/eval/stress_test.py $RUN 16 0.12
check watch     env ${GIF_GL:+MUJOCO_GL=$GIF_GL} $PY tools/watch_policy.py $RUN --vx 0.12   # walk.gif + posture.json (flat)
echo "evals end $(date -u +%H:%M:%S)" >> $T
cat $T
