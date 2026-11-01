#!/usr/bin/env bash
# Run once from the repository in a cluster login shell. The API key is entered
# into W&B's login prompt and never written to this repository.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
command -v uvx >/dev/null || { echo 'uvx is required on the cluster login node' >&2; exit 1; }

read -r -p 'Your W&B project name: ' project
[[ -n "${project//[[:space:]]/}" ]] || { echo 'Project name cannot be empty' >&2; exit 1; }

echo 'W&B will prompt you for your personal API key.'
uvx --from 'wandb==0.22.2' wandb login

umask 077
printf '%s\n' "$project" > .wandb-project
echo 'W&B login configured and project saved in .wandb-project (ignored by Git).'
