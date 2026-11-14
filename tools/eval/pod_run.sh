#!/bin/bash
# Run 19: usage  pod_run.sh <name> <duty> <cache_suffix> [extra --set KEY=VALUE ...]
# (run 21+: extra args go straight to train_walk.py)
NAME=$1; DUTY=$2; SUF=$3; shift 3
cd /workspace/red-panda-bot
F='^(WARNING|W[0-9]{4}|I[0-9]{4})|solver iterations|warn_overflow|^Module '
TT=/workspace/${NAME}_times.txt
echo "script start $(date -u +%H:%M:%S)" > $TT
source /workspace/venv/bin/activate
SP=$(python -c "import site; print(site.getsitepackages()[0])")
SKIP='./nvidia ./jedi ./OpenGL ./mujoco_playground/external_deps'
# --- Import fix: small files in one compressed archive on the volume (built once, by
# whichever pod gets the lock); big files and unused folders stay on the volume as links.
if mkdir /workspace/fastlib.lock 2>/dev/null; then
  if [ ! -f /workspace/fastlib.done ]; then
    t=$(date +%s); cd "$SP"
    find . \( -path ./nvidia -o -path ./jedi -o -path ./OpenGL -o -path ./mujoco_playground/external_deps \) -prune -o -type f -size -20M -print > /workspace/fastlib_files.txt
    find . \( -path ./nvidia -o -path ./jedi -o -path ./OpenGL -o -path ./mujoco_playground/external_deps \) -prune -o -type f -size +20M -print > /workspace/fastlib_big.txt
    MB=$(tr '\n' '\0' < /workspace/fastlib_files.txt | du -cm --files0-from=- | tail -1 | cut -f1)
    echo "small files: $(wc -l < /workspace/fastlib_files.txt), ${MB} MB; big files: $(wc -l < /workspace/fastlib_big.txt)" >> $TT
    if [ "$MB" -lt 4500 ]; then
      if command -v zstd > /dev/null; then Z="zstd -T0 -3"; else Z="gzip -1"; fi
      tar -I "$Z" -cf /workspace/fastlib.tar.part -T /workspace/fastlib_files.txt && mv /workspace/fastlib.tar.part /workspace/fastlib.tar && echo "$Z" > /workspace/fastlib.done
      echo "one-time: archive built with '$Z' in $(( $(date +%s) - t )) s, $(du -sm /workspace/fastlib.tar | cut -f1) MB" >> $TT
    else
      echo "small files too big (${MB} MB), skipping the import fix" >> $TT
    fi
    cd /workspace/red-panda-bot
  fi
else
  for i in $(seq 1 60); do [ -f /workspace/fastlib.done ] && break; sleep 10; done   # other pod is building it
fi
if [ -f /workspace/fastlib.done ]; then
  t=$(date +%s); L=/root/fastlib; mkdir -p $L
  tar -I "$(cat /workspace/fastlib.done) -d" -xf /workspace/fastlib.tar -C $L
  while read -r f; do mkdir -p "$L/$(dirname "$f")"; ln -sf "$SP/${f#./}" "$L/${f#./}"; done < /workspace/fastlib_big.txt
  for d in $SKIP; do mkdir -p "$L/$(dirname "$d")"; ln -sfn "$SP/${d#./}" "$L/${d#./}"; done
  echo "local packages ready in $(( $(date +%s) - t )) s" >> $TT
  export PYTHONPATH=$L
  if python -c "import jax, mujoco, warp, brax, mujoco_playground; from mujoco import mjx; from brax.training.agents.ppo import train; print('devices', jax.devices()); print('playground from', mujoco_playground.__file__)" >> $TT 2>&1; then
    echo "import test with local packages: OK" >> $TT
  else
    echo "import test FAILED -> falling back to the volume's packages" >> $TT; unset PYTHONPATH
  fi
fi
case "$*" in *terrain_config.enable=true*)   # run 22a: JAX takes 75% of the GPU up front;
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70;;    # leave Warp room for its terrain contact buffers
esac
export WARP_CACHE_PATH=/workspace/warp_cache$SUF          # separate caches per pod: no shared writes
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 > /workspace/${NAME}_gpu.csv &
SMI=$!
python tools/train_walk.py --env redpanda --name $NAME --set gait_config.duty=$DUTY "$@" \
  --compile_cache /workspace/jax_cache$SUF 2>&1 | grep --line-buffered -v -E "$F" > /workspace/${NAME}.log
kill $SMI; cp /workspace/${NAME}_gpu.csv runs/$NAME/gpu.csv
echo "training end $(date -u +%H:%M:%S)" >> $TT
echo ALL_DONE >> /workspace/${NAME}.log
