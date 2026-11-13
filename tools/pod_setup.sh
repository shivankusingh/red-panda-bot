#!/bin/bash
# Run this ON the GPU machine (RunPod pod) to get ready for training.
# Puts the Python environment and the repo on /workspace (the network volume),
# so the next session skips the install and just pulls the latest code.
#
#   bash pod_setup.sh
set -e
cd /workspace

# 1. Python environment with the same package versions as the Mac (+ CUDA JAX).
if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q "jax[cuda12]==0.12.0.1" mujoco==4.0.1 mujoco-mjx==4.0.1 \
  warp-lang==1.19.0 brax==0.14.2 playground==0.2.0

# 2. The code (public repo).
if [ ! -d red-panda-bot ]; then
  git clone https://github.com/shivankusingh/red-panda-bot.git
else
  git -C red-panda-bot pull
fi

# 3. Check the GPU is visible to JAX.
python -c "import jax; print('JAX devices:', jax.devices())"
echo "Ready. Next: cd /workspace/red-panda-bot && source /workspace/venv/bin/activate"
