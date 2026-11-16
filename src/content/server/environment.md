---
title: "Software Environment"
description: "OS, NVIDIA driver and CUDA versions, and how to set up Python environments."
section: laal
order: 7
date: "2026-11-16"
---

## System software

| | `laal1` | `laal2` | `laal3` |
| --- | --- | --- | --- |
| OS | Ubuntu 22.04 | Ubuntu 22.04 | Ubuntu 22.04 |
| NVIDIA driver | 580 (CUDA 13.0) | 580 (CUDA 13.0) | 580 (CUDA 13.0) |
| CUDA toolkits in `/usr/local` | 12.1, 12.2, 12.8, 13.0 | 11.7, 12.2, 13.0 | 12.8, 13.0 |

- **You usually don't need a system CUDA toolkit.** PyTorch and JAX wheels ship their own CUDA runtime, and the driver supports CUDA 13.0 and everything older.
- **To compile CUDA extensions** (e.g. flash-attention from source), point to a toolkit that exists on that node:

  ```bash
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=$CUDA_HOME/bin:$PATH
  ```

- **`laal3` (Blackwell) needs builds made for CUDA 12.8 or newer**, e.g. PyTorch `cu128` wheels or newer. Older builds fail there with "no kernel image is available".

## Python environments

Install everything in your home directory. `/home` is visible from every node, so the same environment works everywhere. Either **conda** or **uv** works.

**conda (Miniconda)**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh        # accept the defaults (installs to ~/miniconda3)
conda create -n myenv python=3.11
```

In an `sbatch` script, activate it like this:

```bash
eval "$(conda shell.bash hook)"
conda activate myenv
```

**uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install torch
```

Environments and package caches get large. Clean them up now and then, as shown in [Storage](/server/storage).

## Docker

Docker is **not available** to regular users. Use conda or uv.

## AI coding tools

Claude Code, Codex CLI, and Gemini CLI can be installed in your own home directory, using Node.js through `nvm`. No `sudo` is needed. Running them on `master` to edit code is fine; launch the actual experiments through Slurm.

## Shared datasets

Check `/home/shared` before downloading a public dataset. See [Storage](/server/storage).
