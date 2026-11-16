---
title: "Servers & GPU Selection"
description: "Understand the login and compute nodes, choose a partition, and check availability."
section: laal
language: en
order: 3
date: "2026-11-16"
---

`master` is the SSH login node for editing and job submission. Slurm allocates `laal1`–`laal3` for experiments. All share `/home`, but node-local disks are not shared.

## Current servers

| Node | GPUs | CPU | RAM | Slurm partition |
| --- | --- | --- | --- | --- |
| `master` | None | Xeon Silver 4510, 12 cores | 512GB | Login only |
| `laal1` | 8 × RTX 3090, 24GB | 2 × Xeon Gold 6326, 64 threads | 512GB | `laal_3090` |
| `laal2` | 10 × RTX A6000, 48GB | 2 × Xeon Silver 4410Y, 48 threads | 768GB | `laal_a6000` |
| `laal3` | 4 × RTX PRO 6000 Blackwell Max-Q, 96GB | EPYC 9654, 192 threads | 640GB | `laal_rtx6000` |

These are recorded configurations, not live availability. Slurm currently counts CPUs as hardware threads.

## Choose a GPU

Start with the memory required **on one GPU**:

| Required VRAM | Candidate partition |
| --- | --- |
| Up to 24GB | `laal_3090` |
| More than 24GB, up to 48GB | `laal_a6000` |
| More than 48GB, up to 96GB | `laal_rtx6000` |

Requesting several GPUs does not automatically combine their memory. Your code must support multi-GPU training. Check [software compatibility](/server/environment) for Blackwell.

If your experiment works on both GPU types, a batch script can list several partitions:

```bash
#SBATCH --partition=laal_a6000,laal_3090
```

Slurm chooses an allocation based on priority and availability. Use [Slurm](/server/slurm) rather than logging directly into compute nodes to claim GPUs.

## Check live availability

```bash
# On master
sinfo
squeue -u "$USER"
sinfo -N -O "NodeList:10,Partition:14,Gres:22,GresUsed:28,CPUsState:16,StateLong:10"
```

`GresUsed` reports allocated GPUs. `CPUsState` reports allocated/idle/other/total CPUs. Allocation and actual utilization are different.

## Planned expansion

LAAL 4 is planned with **one GPU initially, expandable to eight**. It is not included in the current-server table. Final resources and the partition will be announced after deployment.

[GSDS storage](/server/gsds-overview) is a separate service under preparation.

