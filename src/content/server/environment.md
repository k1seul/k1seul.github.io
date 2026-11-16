---
title: "Setting Up Your Environment"
description: "Create a personal Python environment and verify GPU support on a compute node."
section: laal
language: en
order: 7
date: "2026-11-16"
---

Install packages in a personal environment. Do not modify system Python or NVIDIA drivers. Choose **conda or uv** below.

## Using conda

If conda is installed, reuse it. Otherwise follow [Miniconda's official installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install) and install in your home directory, normally `~/miniconda3`.

```bash
# On master
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n myenv python=3.11
conda activate myenv
```

Use your project's `requirements.txt` or `environment.yml` first. For PyTorch, choose a suitable Linux/Pip/CUDA combination in the [official installer](https://pytorch.org/get-started/locally/).

A batch job starts a fresh shell, so activate its environment in the script:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myenv
python -u train.py
```

## Using uv

Install uv in your account using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/) if needed.

```bash
# In your project directory on master
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

In a batch script, activate `.venv/bin/activate` from the same project directory. If no requirements file exists, install the packages your project needs.

## Verify GPU access

There is no GPU on master. Test inside a compute-node allocation:

```bash
# Request resources on master
launch-shell 1 30 laal_3090 16
# Once the compute-node shell opens
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
exit
```

The final Python output should be `True`. If using uv, substitute its activation command.

## CUDA and Blackwell compatibility

The recorded software configuration is below. Maintenance may change it; check `nvidia-smi` and, if needed, `nvcc --version` on the allocated node.

| Item | laal1 | laal2 | laal3 |
| --- | --- | --- | --- |
| OS | Ubuntu 22.04 | Ubuntu 22.04 | Ubuntu 22.04 |
| Driver family | 580 | 580 | 580 |
| CUDA toolkits in /usr/local | 12.1, 12.2, 12.8, 13.0 | 11.7, 12.2, 13.0 | 12.8, 13.0 |

The CUDA version displayed by `nvidia-smi` describes driver support, not necessarily your Python environment's runtime. PyTorch wheels often include a runtime, so a system toolkit is not always required just to run a model.

Blackwell on laal3 needs a compatible build. An older CUDA/PyTorch environment may fail; check [official PyTorch versions](https://pytorch.org/get-started/previous-versions/) for a supported combination. A shared home directory does not guarantee compatibility across GPUs or compiled extensions.

Only when compiling a CUDA extension, select an installed toolkit compatible with the project:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

## Other tools

Regular users do not have Docker access. Use conda/uv, or contact an admin about other environments. Personal development and AI coding tools can be installed in your home. Editing on the login node is fine; experiments launched by those tools must still run through Slurm.

Clean up unused environments and caches as described in [Storage](/server/storage).

