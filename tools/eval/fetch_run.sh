#!/bin/bash
# Wait for one pod's run to finish, then copy its small results back to the Mac.
#
#   bash tools/eval/fetch_run.sh rp_run25a 213.0.0.1 12345     (pod SSH host + port)
#
# One watcher per pod, so each pod can be stopped as soon as its own results are home
# (run 24: one stalled copy kept both pods idle ~25 min). Copies params.pkl, the
# training logs, eval/ (from run_evals.sh), posture.json and walk.gif; never the
# ~85 MB checkpoints or the big raw log.
RUN=$1; HOST=$2; PORT=$3
cd "$(dirname "$0")/../.."   # repo root
O="-o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
REMOTE=/workspace/red-panda-bot/runs/$RUN
for i in $(seq 1 180); do   # up to 1 hour
  out=$(ssh $O -p $PORT root@$HOST "grep -hE 'ALL_DONE|Traceback' /workspace/$RUN.log 2>/dev/null | tail -1" < /dev/null 2>/dev/null)
  if [ -n "$out" ]; then
    echo "$RUN: $out"
    mkdir -p runs/$RUN/eval
    for f in params.pkl progress.csv config.json timing.json gpu.csv posture.json walk.gif; do
      scp $O -q -P $PORT root@$HOST:$REMOTE/$f runs/$RUN/ 2>/dev/null
    done
    scp $O -q -P $PORT "root@$HOST:$REMOTE/eval/*" runs/$RUN/eval/ 2>/dev/null
    scp $O -q -P $PORT root@$HOST:/workspace/${RUN}_times.txt root@$HOST:/workspace/$RUN.log runs/$RUN/ 2>/dev/null
    echo "$RUN copied: $(ls runs/$RUN | tr '\n' ' ') | eval: $(ls runs/$RUN/eval | tr '\n' ' ')"
    exit 0
  fi
  sleep 20
done
echo "$RUN: timed out waiting"; exit 1
