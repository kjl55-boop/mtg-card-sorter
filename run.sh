#!/usr/bin/env bash
set -euo pipefail

# change to project dir
cd "$HOME/mtg-card-sorter"

# activate virtualenv if present
if [ -f "venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
else
  echo "Warning: venv not found at venv/bin/activate; continuing without virtualenv"
fi

# update code
git pull origin main

# fetch LFS objects for the current checkout (no-op if not using LFS)
if command -v git-lfs >/dev/null 2>&1; then
  echo "Fetching Git LFS objects..."
  git lfs pull
else
  echo "git-lfs not installed. Install git-lfs to fetch large files (cards.db, descriptors)."
fi

# optional DB verification (uncomment to enable)
# python - <<'PY'
# import os, sys, sqlite3
# db = "data/scryfall_db/cards.db"
# if not os.path.exists(db):
#     print("DB missing:", db); sys.exit(2)
# try:
#     conn = sqlite3.connect(db)
#     conn.execute("SELECT name FROM sqlite_master LIMIT 1")
#     conn.close()
#     print("DB check OK")
# except Exception as e:
#     print("DB open failed:", e); sys.exit(3)
# PY

# start the inspector
python -m app.run_inspector
