---
title: "Servers & Hardware"
description: "The login node, the three GPU nodes, and which partition to use."
section: laal
order: 3
date: "2026-11-16"
---

## How the cluster is organized

- **`master`** is the login node. You SSH into it, edit code, and submit jobs. It also holds `/home`, which every node sees over the network.
- **`laal1`, `laal2`, `laal3`** are GPU nodes. You never log in to them directly; Slurm runs your job there.

## Nodes

| Node | GPUs | CPU | RAM | Slurm partition |
| --- | --- | --- | --- | --- |
| `master` | none | Xeon Silver 4510 (12 cores) | 512 GB | (login only) |
| `laal1` | 8 × RTX 3090 (24 GB) | 2 × Xeon Gold 6326 (64 threads) | 512 GB | `laal_3090` (default) |
| `laal2` | 10 × RTX A6000 (48 GB) | 2 × Xeon Silver 4410Y (48 threads) | 768 GB | `laal_a6000` |
| `laal3` | 4 × RTX PRO 6000 Blackwell Max-Q (96 GB) | AMD EPYC 9654 (192 threads) | 640 GB | `laal_rtx6000` |

Slurm counts CPUs as hardware threads, so `--cpus-per-task=4` means 4 threads.

## Which partition should I use?

| Your model / batch needs | Use |
| --- | --- |
| up to ~24 GB of GPU memory | `laal_3090` |
| up to ~48 GB | `laal_a6000` |
| more than 48 GB | `laal_rtx6000` |

`laal_rtx6000` has the fewest GPUs and usually the longest queue, so only use it when you really need 96 GB.

If your job fits on more than one GPU type, list several partitions and Slurm picks whichever frees up first:

```bash
#SBATCH --partition=laal_a6000,laal_3090
```

## Checking live status

```bash
sinfo -N -O "NodeList:10,Partition:14,Gres:22,GresUsed:28,CPUsState:16,StateLong:10"   # free GPUs/CPUs per node
squeue                                                                                 # everyone's jobs
squeue -u $USER                                                                        # your jobs
```

In `GresUsed`, `gpu:a6000:9(IDX:0-8)` means 9 of the 10 GPUs on that node are in use. In `CPUsState`, the numbers are allocated/idle/other/total.
