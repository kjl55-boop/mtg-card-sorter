#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/mtg-card-sorter"

if [ -f "venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
else
  echo "Warning: venv not found at venv/bin/activate; continuing without virtualenv"
fi

git pull origin main

if command -v git-lfs >/dev/null 2>&1; then
  echo "Fetching Git LFS objects..."
  git lfs pull
else
  echo "git-lfs not installed. Install git-lfs to fetch large files (cards.db, descriptors)."
fi

python -m app.run_inspector
