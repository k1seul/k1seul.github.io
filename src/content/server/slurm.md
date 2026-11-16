---
title: "Running Jobs with Slurm"
description: "Use interactive shells or batch jobs, read logs, and release your resources."
section: laal
language: en
order: 4
date: "2026-11-16"
---

Slurm schedules shared GPUs, CPUs, and memory. Unless noted otherwise, run submission and status commands on **master**.

| What you want | Use |
| --- | --- |
| Debug or run a short test | `launch-shell` |
| Run training after closing your terminal | `sbatch` |

## Interactive jobs

```bash
launch-shell <GPU-count> [minutes] [partition] [memory-GB]
```

| Argument | Default | Notes |
| --- | --- | --- |
| GPU count | Required | 0–4 |
| Time | 60 minutes | Up to 480 minutes |
| Partition | `laal_3090` | [GPU choices](/server/hardware) |
| CPU memory | 10GB per GPU; 10GB for zero GPUs | Separate from GPU VRAM |

The helper allocates `2 × GPU-count + 2` CPUs. For one A6000 for two hours with 40GB RAM:

```bash
launch-shell 1 120 laal_a6000 40
# Once the compute-node shell opens
nvidia-smi
# Activate your Python environment, then run
python train.py
exit
```

`exit` releases the allocation. The shell ends when its time limit expires; use a batch job for long experiments.

## Batch jobs

Save the following as `run.sh` in your experiment directory. [Create an environment](/server/environment) first and replace `myenv` with its name.

```bash
#!/bin/bash
#SBATCH --job-name=my_exp
#SBATCH --partition=laal_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myenv
python -u train.py
```

This assumes Miniconda's default install path. For uv, replace the environment setup lines with `source .venv/bin/activate`. Keep all `#SBATCH` directives before shell commands. Shell variables such as `$USER` are not expanded inside those directives.

```bash
# On master, in your experiment directory
mkdir -p slurm_logs
sbatch run.sh
```

`Submitted batch job 405` confirms submission, not that execution has started. Create the log directory **before** submission: Slurm does not create it. The default working directory is where you submit the job.

## Status, logs, and cancellation

```bash
squeue -u "$USER"
tail -f slurm_logs/my_exp-405.out
scancel 405
```

Replace `405` with your job ID. Ctrl+C exits `tail -f` without stopping training. `scancel` stops the job. Use `scancel -u "$USER"` only when you intend to cancel **all your jobs**.

Inspect a GPU inside an existing allocation:

```bash
srun --jobid=405 --overlap --pty nvtop
# If nvtop is unavailable
srun --jobid=405 --overlap nvidia-smi
```

Check completed or failed jobs with:

```bash
sacct -j 405 -o JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

Memory readings depend on accounting settings and may be empty. MaxRSS is a per-step maximum process value, not necessarily the job's total RAM use.

## How much to request

| Resource | Starting point | Adjust based on |
| --- | --- | --- |
| CPU | 4 per GPU | Dataloader use |
| CPU RAM | 30–60GB per GPU | Measured use after the first run |
| GPU count | What the code will actually use | Multi-GPU support in your code |
| Time | Estimated runtime plus reasonable margin | Jobs stop at the limit |

`--mem` requests **CPU RAM**, not GPU VRAM. Current per-node request limits are 500G on `laal1`/`laal3` and 566G on `laal2`; check `scontrol show partition` and admin notices for changes.

Low GPU utilization can come from I/O, preprocessing, CPU limits, or synchronization. More CPUs alone may not help.

## Why is my job pending?

| Reason | Meaning / next step |
| --- | --- |
| `Resources` | Waiting for all requested resources. Check CPU and RAM requests too. |
| `Priority` | Higher-priority jobs are ahead; not necessarily submission order. |
| `Dependency` | Waiting for an earlier job's completion condition. |
| `ReqNodeNotAvail` | Required nodes are unavailable. Check notices or contact an admin. |

Use `sinfo` to check other partitions. Send the job ID if an unfamiliar time-limit or QOS reason appears.

## Dependencies and arrays

Run evaluation only after job 405 succeeds:

```bash
sbatch --dependency=afterok:405 test.sh
```

For sweeps, add an array directive before shell commands in your batch script. `%4` caps concurrent tasks at four. Activate your environment before invoking the Python script.

```bash
#SBATCH --array=0-99%4
# After your environment setup
python sweep.py --idx "$SLURM_ARRAY_TASK_ID"
```

See the [official sbatch documentation](https://slurm.schedmd.com/sbatch.html) for more options.

