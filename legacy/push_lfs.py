#!/usr/bin/env python3
"""
tools/push_lfs.py

Usage:
  python tools/push_lfs.py \
    --remote origin \
    --branch main \
    --paths data/scryfall_db/cards.db data/scryfall_db/descriptors/*.npz \
    --message "Add scryfall_db via Git LFS"

This script runs git-lfs install, git lfs track on the provided patterns,
adds .gitattributes and the files, then commits and pushes to remote/branch.

Caveats:
- Requires git and git-lfs available in PATH.
- Runs git commands in the current working directory (repo root).
- If files are already committed to normal Git history, consider using Git LFS migration tooling instead.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

def run(cmd, check=True, capture=False):
    print(f"> {cmd}")
    proc = subprocess.run(shlex.split(cmd), check=(check and True), capture_output=capture, text=True)
    if capture:
        return proc.stdout.strip()
    return None

def ensure_git_lfs():
    try:
        out = run("git lfs version", check=True, capture=True)
        print("git-lfs available:", out)
    except subprocess.CalledProcessError:
        print("git-lfs does not appear to be installed or not in PATH.", file=sys.stderr)
        sys.exit(2)
    # install for this repo (no-op if already)
    run("git lfs install")

def git_lfs_track(patterns):
    for p in patterns:
        run(f'git lfs track "{p}"')
    run("git add .gitattributes")

def add_and_commit(paths, message):
    # Expand globs in shell-safe manner
    added = []
    for p in paths:
        # Allow quotes/wildcards; use Path.glob for local expansion when possible
        if "*" in p or "?" in p or "[" in p:
            for match in Path().glob(p):
                run(f'git add "{match}"')
                added.append(str(match))
        else:
            run(f'git add "{p}"')
            added.append(p)
    if not added:
        print("No files matched to add. Ensure the paths are correct.", file=sys.stderr)
    # Commit if there are staged changes
    try:
        status = run("git rev-parse --abbrev-ref HEAD", capture=True)
        print("Current branch:", status)
    except Exception:
        pass
    try:
        run(f'git commit -m "{message}"')
    except subprocess.CalledProcessError:
        # No changes to commit
        print("Nothing to commit (maybe files already added).")

def push(remote, branch):
    run(f"git push {remote} {branch}")
    # ensure LFS objects are pushed
    run("git lfs push --all {remote}".format(remote=remote))

def parse_args():
    p = argparse.ArgumentParser(description="Track and push large files with Git LFS")
    p.add_argument("--remote", default="origin", help="Git remote name (default: origin)")
    p.add_argument("--branch", default=None, help="Branch to push (default: current branch)")
    p.add_argument("--paths", nargs="+", required=True,
                   help="Paths or glob patterns to add to LFS and commit (e.g. data/scryfall_db/cards.db data/scryfall_db/descriptors/*.npz)")
    p.add_argument("--message", default="Add large files via Git LFS", help="Commit message")
    return p.parse_args()

def get_current_branch():
    try:
        out = run("git rev-parse --abbrev-ref HEAD", capture=True)
        return out
    except Exception:
        return None

def main():
    args = parse_args()
    ensure_git_lfs()

    # track patterns via LFS
    git_lfs_track(args.paths)

    # determine branch
    branch = args.branch or get_current_branch() or "main"
    print("Using branch:", branch)

    # add & commit files
    add_and_commit(args.paths, args.message)

    # push
    try:
        push(args.remote, branch)
        print("Push complete.")
    except subprocess.CalledProcessError as e:
        print("Push failed:", e, file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
