#!/usr/bin/env bash
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# Codespaces mount /tmp on a separate large disk; keep the ~3 GB wheel cache off the workspace disk.
export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
git config core.fileMode false
cp -n .env.example .env || true
make install
