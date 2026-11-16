---
title: "Troubleshooting"
description: "Check connection issues, pending jobs, memory errors, and storage problems."
section: laal
language: en
order: 8
date: "2026-11-16"
---

Keep the error text and job ID first. If the checks below do not help, send the [problem report details](/server/administration#reporting-a-problem).

## I cannot connect

| Symptom | What to check |
| --- | --- |
| Timeout | WARP organization/status off campus, address, port, maintenance notices |
| Permission denied | Username, password, registered SSH key |
| Host key changed | Stop and confirm the fingerprint with the admin before reconnecting |

See [Off-Campus Access](/server/remote-access) and [SSH setup](/server/getting-started).

## My job is still pending

```bash
# On master
squeue -u "$USER"
sinfo
```

With `Resources`, CPU and RAM must be available as well as GPUs. `Priority` means waiting on scheduler priority. See [Slurm](/server/slurm) for other reasons.

## My experiment stopped

```bash
sacct -j <JOBID> -o JobID,State,ExitCode,Elapsed,MaxRSS
```

| State or log | Meaning | Next step |
| --- | --- | --- |
| `OUT_OF_MEMORY` | CPU RAM limit may be exceeded | Check recorded use and dataloaders, adjust --mem |
| `TIMEOUT` | Time limit reached | Check checkpoint resume support and --time |
| `CUDA out of memory` | GPU VRAM exhausted | Smaller batches, mixed precision, or a larger-memory GPU |
| Immediate `FAILED` | Environment, paths, code, or another error | Read the first error in the job log |

MaxRSS may be absent or appear only for some steps. Check logs as well; CPU RAM and GPU VRAM are separate resources.

## GPUs are missing or the run is slow

Use `hostname` to check whether you are on master. Inside an allocation, check `nvidia-smi` and your Python environment. A Blackwell `no kernel image is available` error may indicate [incompatible software](/server/environment).

Low utilization can come from data loading, preprocessing, CPUs, or synchronization. Consider [local scratch](/server/storage) if shared storage is a bottleneck.

## No space left on device

Check the **actual output directory**:

```bash
df -h .
df -i .
du -sh .
```

Capacity, inodes, quotas, or temporary directories can be responsible. Remove only your own unnecessary files. Do not delete shared data; tell an admin if several users are affected.

## I deleted a file

There is no automatic backup. Stop extra writes and cleanup, then contact an admin immediately. Another copy may exist, but recovery is not guaranteed.

## I cannot see the new GSDS storage

It is not open yet. Check the [service status](/server/gsds-overview) and keep using existing GSDS space until launch.

