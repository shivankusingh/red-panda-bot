#!/bin/bash
# Run 19: usage  pod_run.sh <name> <duty> <cache_suffix> [extra --set KEY=VALUE ...]
# (run 21+: extra args go straight to train_walk.py)
NAME=$1; DUTY=$2; SUF=$3; shift 3
cd /workspace/red-panda-bot
F='^(WARNING|W[0-9]{4}|I[0-9]{4})|solver iterations|warn_overflow|^Module '
TT=/workspace/${NAME}_times.txt
echo "script start $(date -u +%H:%M:%S)" > $TT
source tools/eval/pod_env.sh   # venv + fast imports (sets PYTHONPATH)
# Run 22a: JAX takes 75% of the GPU up front; leave Warp room for its terrain contact
# buffers. Always on since terrain became the default (run 24a); flat runs fit too.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export WARP_CACHE_PATH=/workspace/warp_cache$SUF          # separate caches per pod: no shared writes
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 > /workspace/${NAME}_gpu.csv &
SMI=$!
# TRAIN_ENV=rear trains rearing (pass "-" as duty; the walking evals are skipped).
if [ "${TRAIN_ENV:-redpanda}" = "rear" ]; then EXTRA=(); EVALS=0; else EXTRA=(--set gait_config.duty=$DUTY); fi
python tools/train_walk.py --env ${TRAIN_ENV:-redpanda} --name $NAME "${EXTRA[@]}" "$@" \
  --compile_cache /workspace/jax_cache$SUF 2>&1 | grep --line-buffered -v -E "$F" > /workspace/${NAME}.log
kill $SMI; cp /workspace/${NAME}_gpu.csv runs/$NAME/gpu.csv
echo "training end $(date -u +%H:%M:%S)" >> $TT
# Run 25+: the measurement suite on the still-warm pod (skip with EVALS=0).
if [ "${EVALS:-1}" = "1" ] && [ -f runs/$NAME/params.pkl ]; then
  unset XLA_PYTHON_CLIENT_MEM_FRACTION
  GIF_GL=egl bash tools/eval/run_evals.sh $NAME >> $TT 2>&1
fi
echo ALL_DONE >> /workspace/${NAME}.log
