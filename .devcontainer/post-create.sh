#!/usr/bin/env bash
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git config core.fileMode false
cp -n .env.example .env || true
make install
# The wheel cache is ~3 GB and only speeds up reinstalls; reclaim it for data.
uv cache prune --ci
