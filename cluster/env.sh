# Sourced by the cluster scripts. cluster/modules.sh is written by hand from the survey: it loads the modules and
# sets PYTHON (3.13+), TensorRT_DIR (10.14+) and CUDA_TOOLKIT_ROOT_DIR (13+).
if [ ! -f cluster/modules.sh ]; then
  echo "cluster/modules.sh is missing: run cluster/survey.sh and write it first (see cluster/README.md)" >&2
  exit 1
fi
source cluster/modules.sh
: "${PYTHON:?cluster/modules.sh must set PYTHON}"
: "${TensorRT_DIR:?cluster/modules.sh must set TensorRT_DIR}"
: "${CUDA_TOOLKIT_ROOT_DIR:?cluster/modules.sh must set CUDA_TOOLKIT_ROOT_DIR}"
export TensorRT_DIR CUDA_TOOLKIT_ROOT_DIR
export HIVEMIND_PYTHON="$PWD/.venv/bin/python"  # the engine converts the FP32 model to FP16 with this
