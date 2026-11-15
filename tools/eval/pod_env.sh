# Sourced (not run) on a RunPod pod: activates the volume venv and sets up fast imports.
#   source tools/eval/pod_env.sh      (from /workspace/red-panda-bot; logs to $TT if set)
# Used by pod_run.sh (training + evals) and for running evals alone on a pod.
TT=${TT:-/dev/stdout}
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
# Headless rendering for the GIF (run_evals.sh): the image has NVIDIA's EGL driver but not
# the libEGL.so.1 loader. ~7 s to install; the container disk is wiped on restart.
if ! ldconfig -p | grep -q "libEGL.so.1"; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1) > /dev/null 2>&1 \
    && echo "installed libegl1 (GIF rendering)" >> $TT || echo "libegl1 install failed: no GIF" >> $TT
fi
