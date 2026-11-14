#!/bin/bash
cd "$(dirname "$0")/../.."   # repo root
# usage: watch_pair.sh "HOST_A PORT_A" "HOST_B PORT_B"   (pod SSH addresses from get-pod)
HA=$1; HB=$2
host() { case $1 in a) echo "$HA";; b) echo "$HB";; esac; }
left="a b"
for i in $(seq 1 100); do
  for x in $left; do
    set -- $(host $x)
    out=$(ssh -o ConnectTimeout=15 -p $2 root@$1 "grep -hE 'ALL_DONE|Traceback|Error' /workspace/rp_run19$x.log 2>/dev/null | tail -1" < /dev/null 2>/dev/null)
    if [ -n "$out" ]; then
      echo "19$x: $out"
      mkdir -p runs/rp_run19$x
      for f in params.pkl progress.csv config.json timing.json gpu.csv; do scp -q -P $2 root@$1:/workspace/red-panda-bot/runs/rp_run19$x/$f runs/rp_run19$x/; done
      scp -q -P $2 root@$1:/workspace/rp_run19${x}_times.txt root@$1:/workspace/rp_run19$x.log runs/rp_run19$x/
      echo "19$x copied: $(ls runs/rp_run19$x | tr '\n' ' ')"
      left=$(echo $left | sed "s/$x//")
    fi
  done
  [ -z "$(echo $left | tr -d ' ')" ] && break
  sleep 20
done
