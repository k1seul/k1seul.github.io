---
title: "Where to Store Files"
description: "Choose shared home or local scratch, check capacity, and keep important copies."
section: laal
language: en
order: 5
date: "2026-11-16"
---

Use shared `/home` for code and results, and available node-local scratch for I/O-heavy temporary data. **LAAL storage has no separate backup.**

## Choose a path

| Path | Purpose | Visible from |
| --- | --- | --- |
| `/home/<username>` | Code, environments, results | All LAAL nodes |
| `/home/shared/<dataset>` | Shared lab datasets | All LAAL nodes |
| `/data/<username>` on laal3 | Fast NVMe scratch | laal3 only |
| `/data/laal2/shared` on laal2 | Limited temporary storage | laal2 only |

The listed laal2 path shares its disk with the OS. Ask the admin before storing large data there. Request a personal scratch directory if yours is missing or inaccessible.

## Check before a run

```bash
# On the node where the work runs
df -h /home
du -sh "$HOME"
du -h --max-depth=1 "$HOME" | sort -h | tail -15
```

`df` shows filesystem capacity; `du` shows directory usage. Large directory scans can take time, so avoid running them repeatedly. When storage errors occur, check the filesystem of the actual output path.

LAAL `/home` currently has no per-user quota. Leave room for checkpoints and remove failed runs, duplicates, and unused checkpoints.

## Use local scratch

Inside a job allocated on that node:

1. Check the node and available local capacity.
2. Copy only the data you need to your own scratch directory.
3. Point experiment inputs and temporary outputs there.
4. Copy retained results to `/home` and verify the copy.
5. Remove only your own temporary files when no longer needed.

Other nodes cannot see those files. Failure or timeout can skip a final copy, so also save important checkpoints to the retained path at appropriate intervals.

## Shared datasets and caches

Check `/home/shared` before downloading a dataset. Add a README with the source and usage instructions for a new dataset under `/home/shared/<dataset_name>`. Ask for permission if needed and do not change other users' data.

Inspect caches before deciding what to clean. There is no need to run every command, and cleaning may require downloads later.

```bash
conda clean --all --dry-run
# Review first; run conda clean --all only if needed
pip cache info
uv cache size
du -sh ~/.cache/huggingface
```

Many small files or frequent simultaneous checkpoints can slow shared HDD storage. Adjust save frequency and use local storage when appropriate.

## RAID is not backup

RAID can tolerate some disk failures, but it is not a separate copy for deleted files, overwrites, or loss of the server. Push code to a remote git repository and keep irreplaceable final checkpoints and results elsewhere too.

[GSDS storage](/server/gsds-overview) is a separate service under preparation. It does not automatically extend LAAL `/home`.

