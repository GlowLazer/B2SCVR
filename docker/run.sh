#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-b2scvr:cu121}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker build -t "${IMAGE}" .
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

docker run --rm -it \
  --gpus all \
  -u "$(id -u):$(id -g)" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE}" "$@"
