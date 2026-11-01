#!/usr/bin/env bash
# Run once per student account on the login node, before opening OnDemand
# JupyterLab. Only the kernel definition is copied; Python stays shared.
set -euo pipefail

P2_VENV=/data/gent/courses/2026/ASIP_E092841/input/P2_venv
SPEC="$P2_VENV/share/jupyter/kernels/asip-p2-doduo/kernel.json"
DEST="$HOME/.local/share/jupyter/kernels/asip-p2-doduo"

test -f "$SPEC" || { echo "Missing shared kernel spec: $SPEC" >&2; exit 1; }
test -x "$P2_VENV/bin/kernel-doduo" || { echo "Missing doduo kernel launcher" >&2; exit 1; }
test -x "$P2_VENV/doduo/bin/python" || { echo "Missing doduo Python environment" >&2; exit 1; }
mkdir -p "$DEST"
install -m 0644 "$SPEC" "$DEST/kernel.json"
echo "Registered ASIP P2 (doduo CPU). Start OnDemand JupyterLab on doduo and select this kernel."
