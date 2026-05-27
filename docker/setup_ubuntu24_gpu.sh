#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (e.g. sudo bash docker/setup_ubuntu24_gpu.sh)"
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "ERROR: /etc/os-release not found"
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "ERROR: This script targets Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}"
  exit 1
fi

echo "==> Updating apt index"
apt-get update

echo "==> Installing prerequisites"
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  ubuntu-drivers-common

echo "==> Installing NVIDIA driver (recommended)"
# If the host already has a working driver, this is effectively a no-op.
ubuntu-drivers autoinstall

echo "==> Installing Docker Engine (official repo)"
install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
fi

ARCH="$(dpkg --print-architecture)"
CODENAME="${VERSION_CODENAME:-noble}"
cat >/etc/apt/sources.list.d/docker.list <<EOF
deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable
EOF

apt-get update
apt-get install -y --no-install-recommends \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

echo "==> Installing NVIDIA Container Toolkit"
distribution=$(. /etc/os-release;echo "${ID}${VERSION_ID}")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
  >/etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit

echo "==> Configuring Docker runtime for NVIDIA"
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "==> Done."
echo "- Reboot is typically required after driver installation."
echo "- After reboot, verify:"
echo "    nvidia-smi"
echo "    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"

