---
title: "Storage"
description: "Where to put code, datasets, and checkpoints, and why nothing is backed up."
section: laal
order: 5
date: "2026-11-16"
---

## Where things go

| Path | Use it for | Visible from | Backed up |
| --- | --- | --- | --- |
| `/home/<username>` | code, environments, results | all nodes | **No** |
| `/home/shared/<dataset>` | public datasets shared by the lab | all nodes | **No** |
| `/data/<username>` on `laal3` | fast local scratch (NVMe) | `laal3` only | **No** |
| `/data/laal2/shared` on `laal2` | local scratch | `laal2` only | **No** |

## Nothing is backed up

`/home` is on a RAID array. RAID survives a broken disk, but it does **not** protect against deleting or overwriting files. If you delete something, it is gone.

- Keep code in git and push it to GitHub.
- Copy results you can't reproduce, like final checkpoints and paper figures, somewhere else too.

## `/home` is shared, so keep it tidy

Everyone's `/home` sits on the same disks, and there are no per-user quotas. When it fills up, **everyone's jobs fail** while saving checkpoints. Please:

- keep only the checkpoints you need (e.g. best and last), and delete failed or abandoned runs
- clear caches from time to time:

```bash
conda clean -a          # conda package cache
pip cache purge         # pip cache
uv cache clean          # uv cache
du -sh ~/.cache/huggingface    # Hugging Face downloads; delete models you no longer use
```

To see what takes up your space (this can take a while on big folders):

```bash
du -h --max-depth=1 ~ | sort -h | tail -15
```

`/home` sits on hard disks shared by everyone. Jobs that write thousands of tiny files, or save checkpoints every few minutes, slow down the whole lab. Write less often, or use local scratch.

## Shared datasets: `/home/shared`

- Before downloading a public dataset (COCO, ADE20K, ...), check whether it's already in `/home/shared`.
- To add one, put it in `/home/shared/<dataset_name>/` with a short `README` saying where it came from.
- Don't delete or change anyone else's data there.

## Local scratch: `/data`

Local disks are faster than `/home` for heavy reading and writing, but only the node they're in can see them.

- **`laal3`**: use `/data/<username>`. If it doesn't exist, ask the admins.
- **`laal2`**: `/data/laal2/shared` lives on the **same disk as the operating system**. Keep it small, because filling it can take the whole node down.
- Copy the results you want to keep back to `/home` at the end of the job, and delete your scratch files when you're done.
