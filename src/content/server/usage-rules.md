---
title: "Usage Rules"
description: "How GPUs and storage are shared, and how to run long jobs."
order: 3
date: "2026-11-16"
---

## GPU allocation

> **TODO:** How GPUs are shared, e.g. reservation sheet or scheduler, maximum GPUs per person, priority before deadlines.

Before launching a job, check which GPUs are free with `nvidia-smi`, then pin your job to them:

```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py
```

## Long-running jobs

Run training inside `tmux` so it survives SSH disconnects:

```bash
tmux new -s train      # start a session
# ... run your job, then detach with Ctrl-b d
tmux attach -t train   # come back later
```

## Storage

> **TODO:** Where to put code vs. datasets vs. checkpoints, quotas, cleanup policy, and whether anything is backed up.

## Etiquette

> **TODO:** Lab-specific rules, e.g. don't leave idle processes holding GPU memory, announce large jobs in the lab channel.
