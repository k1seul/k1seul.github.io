---
title: "FAQ & Troubleshooting"
description: "Common problems and how to fix them."
section: laal
order: 8
date: "2026-11-16"
---

## I can't connect

- **Off campus?** Cloudflare WARP must be installed and connected. See [Off-Campus Access](/server/remote-access).
- Check that you're using the server address the admin sent you.
- `Permission denied`: check your username, your password, and that your key is in `~/.ssh/authorized_keys`.

## My job has been pending for a long time

Check the reason in `squeue -u $USER`.

- `Resources`: the node doesn't have enough free GPUs, CPUs, or memory. Check `sinfo` for a less busy partition, or lower your `--cpus-per-task` and `--mem`.
- `Priority`: other jobs are ahead of you in the queue.

See [Why is my job pending?](/server/slurm#why-is-my-job-pending)

## My job died

| What you see | Likely cause | Fix |
| --- | --- | --- |
| State `OUT_OF_MEMORY` | used more CPU RAM than `--mem` | raise `--mem` (check `MaxRSS` with `sacct`) |
| State `TIMEOUT` | hit the `--time` limit | raise `--time`, and save checkpoints so you can resume |
| `CUDA out of memory` in the log | GPU memory is full | smaller batch size, mixed precision, or a GPU with more memory |
| Fails within seconds | usually a script error | read the log in `slurm_logs/` |

## My job is much slower than expected

- Look at GPU utilization: `srun --jobid=<JOBID> --overlap --pty nvtop`.
- If the GPU is mostly idle, the dataloader is the bottleneck. Tune `num_workers`, or copy the data to local scratch (`/data` on `laal3`).

## "No space left on device"

`/home` is full. Delete what you don't need (see [Storage](/server/storage)) and tell the admins.

## I deleted something by mistake

There is no backup. Tell the admins right away anyway; sometimes a copy still exists elsewhere, but there are no guarantees.

## How do I report a problem?

DM Hanseul Kim with the details listed in [Administration](/server/administration#reporting-a-problem).
