#!/usr/bin/env bash
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git config core.fileMode false
cp -n .env.example .env || true
make install
