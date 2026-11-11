#!/usr/bin/env bash
# Run a command with the personal GitHub account (ljofreflor), isolated in docker.
#   tools/gh-docker/ghd.sh 'gh auth login --hostname github.com --git-protocol https'
#   tools/gh-docker/ghd.sh 'git push origin develop'
#   tools/gh-docker/ghd.sh 'gh run list'
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
image=lme4-gh
docker image inspect "$image" >/dev/null 2>&1 || docker build -q -t "$image" "$here" >/dev/null
MSYS_NO_PATHCONV=1 docker run --rm -i \
  -v lme4-gh-ljofreflor:/root/.config/gh \
  -v "$repo":/work \
  "$image" "$*"
