---
title: "Running Jobs with Slurm"
description: "Interactive shells with launch-shell, batch jobs with sbatch, and how much CPU and memory to ask for."
section: laal
order: 4
date: "2026-11-16"
---

All GPU work goes through **Slurm**. There are two ways to use it:

| You want to... | Use |
| --- | --- |
| code, debug, try things interactively | `launch-shell` (interactive shell on a GPU node) |
| run an experiment and check back later | `sbatch` (batch job) |

## Interactive: `launch-shell`

```bash
launch-shell <#GPUs> [minutes] [partition] [memory in GB]
```

| Argument | Default | Limits |
| --- | --- | --- |
| `#GPUs` | (required) | 0 to 4 |
| `minutes` | 60 | up to 480 (8 hours) |
| `partition` | `laal_3090` | any partition |
| `memory` | 10 GB per GPU (10 GB with no GPU) | |

You also get `2 × GPUs + 2` CPUs. For example, one A6000 for two hours with 40 GB of RAM:

```bash
launch-shell 1 120 laal_a6000 40
```

The GPU stays reserved until you `exit` or the time runs out, so **exit as soon as you're done**.

## Batch jobs: `sbatch`

Save this as `run.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=my_exp
#SBATCH --partition=laal_a6000          # laal_3090 | laal_a6000 | laal_rtx6000
#SBATCH --gres=gpu:1                    # number of GPUs
#SBATCH --cpus-per-task=4               # see the guidelines below
#SBATCH --mem=48G                       # CPU RAM, not GPU memory
#SBATCH --time=1-00:00:00               # D-HH:MM:SS
#SBATCH --output=slurm_logs/%x-%j.out   # %x = job name, %j = job ID

eval "$(conda shell.bash hook)"
conda activate myenv

python train.py
```

Submit it:

```bash
mkdir -p slurm_logs     # Slurm does not create the log folder for you
sbatch run.sh
```

## Watching and cancelling jobs

```bash
squeue -u $USER                      # your running and pending jobs
scancel <JOBID>                      # cancel one job
scancel -u $USER                     # cancel all of your jobs
sacct -X -S today -o JobID,JobName%20,Partition,State,Elapsed,ExitCode   # what happened today
```

A more readable `squeue`: add these lines to `~/.profile`, then use `sq`.

```bash
export SQUEUE_FORMAT="%8i %20j %4t %12u %14P %5D %20R %4C %13b %8m %10M"
alias sq='squeue'
```

To see how busy the GPU of a running job is, open a second shell inside the job:

```bash
srun --jobid=<JOBID> --overlap --pty nvtop     # or: nvidia-smi
```

## How much CPU and memory to ask for

These numbers come from the lab's own usage data. Typical jobs use **1–3 CPU cores per GPU**, and half of them use **under 6 GB of RAM**. Asking for much more than you use blocks other people's jobs even when GPUs are free.

| Resource | Ask for | Notes |
| --- | --- | --- |
| CPUs | **4 per GPU** | Go higher only if your dataloader really needs it. `laal2` has 48 CPUs for 10 GPUs, so more than 4 per GPU there leaves GPUs idle. |
| Memory | **30–60 GB per GPU** | Check your real usage after the first run (below) and adjust. |
| Time | a realistic estimate | The default is 1 day. Shorter, accurate limits let the scheduler start your job earlier by fitting it into gaps. |

The most you can request on one node is `--mem=500G` on `laal1` and `laal3`, and `--mem=566G` on `laal2`.

Check what a finished job actually used:

```bash
sacct -j <JOBID> -o JobID,Elapsed,MaxRSS,State
```

If the GPU is above ~80% utilization, the job is GPU-bound and more CPUs or RAM won't make it faster. If it is low, the dataloader is usually the bottleneck. Tuning `num_workers` helps more than asking for more CPUs.

## Chaining jobs (dependencies)

Run `test.sh` only after the training job finishes:

```bash
sbatch run.sh                                  # prints: Submitted batch job 405
sbatch --dependency=afterok:405 test.sh        # starts only if 405 succeeded
sbatch --dependency=afterany:405 test.sh       # starts however 405 ended
```

## Many small runs (job arrays)

For sweeps, use one array instead of hundreds of separate jobs. `%4` caps how many run at the same time, so you leave GPUs for others:

```bash
#SBATCH --array=0-99%4
python sweep.py --idx $SLURM_ARRAY_TASK_ID
```

## Why is my job pending?

`squeue -u $USER` shows the reason in the last column:

| Reason | Meaning |
| --- | --- |
| `Resources` | Waiting for enough free GPUs, CPUs, or memory on the node |
| `Priority` | Other jobs submitted earlier are ahead of you (first come, first served) |
| `Dependency` | Waiting for the job it depends on |
| `ReqNodeNotAvail` | The node is down or in maintenance |

If you're stuck on `Resources`, check `sinfo` (see [Servers & Hardware](/server/hardware)) for another partition with free GPUs, or lower your CPU and memory request.
