#!/usr/bin/env bash
set -eo pipefail

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  set +u
  source /opt/conda/etc/profile.d/conda.sh
  conda activate b2scvr
  set -u
fi

exec "$@"

