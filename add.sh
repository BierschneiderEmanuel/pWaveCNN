#!/usr/bin/env bash
# Add all files and push to the remote GitHub repo.
# Usage: ./add.sh ["commit message"]
set -euo pipefail

REMOTE_URL="git@github.com:BierschneiderEmanuel/pWaveCnn.git"
BRANCH="main"
MSG="${1:-Update $(date '+%Y-%m-%d %H:%M:%S')}"

cd "$(dirname "$0")"

# Initialize repo if needed
if [ ! -d .git ]; then
    git init
    git branch -M "$BRANCH"
fi

# Add or fix the remote
if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
fi

# Warn about files larger than GitHub's 100 MB limit
# (only files git would actually add, so .gitignore'd paths like .venv are skipped)
large=$(git ls-files -z --cached --others --exclude-standard \
    | xargs -0 -r find 2>/dev/null -maxdepth 0 -type f -size +100M -print || true)
if [ -n "$large" ]; then
    echo "WARNING: files over 100 MB (GitHub will reject them; add to .gitignore):"
    echo "$large"
    exit 1
fi

git add -A

if git diff --cached --quiet; then
    echo "Nothing new to commit."
else
    git commit -m "$MSG"
fi

git branch -M "$BRANCH"

# Merge remote history if the remote branch already exists
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git pull --no-rebase origin "$BRANCH" --allow-unrelated-histories --no-edit
fi

git push -u origin "$BRANCH"
