---
title: "Usage Rules"
description: "A few rules that keep the servers fast and fair for everyone."
section: laal
order: 6
date: "2026-11-16"
---

The servers are shared by about 40 people. These rules keep them usable for all of us.

## 1. Run experiments through Slurm, not on `master`

`master` is the login node. Editing, git, small scripts, and AI coding tools are fine there. **Training, evaluation, preprocessing, and anything that runs for a long time are not.** Put that work in `launch-shell` or `sbatch`.

## 2. Ask for what you use

Start with **4 CPUs and 30–60 GB of RAM per GPU**, then adjust to what your job actually uses. See [Running Jobs with Slurm](/server/slurm#how-much-cpu-and-memory-to-ask-for). Over-asking keeps GPUs idle that someone else could use.

## 3. Don't hold GPUs you aren't using

- Don't keep a GPU job alive just to wait for work, e.g. a script that loops on `sleep`. Submit the job when the work is ready.
- `exit` your `launch-shell` when you're done.
- Cancel jobs you no longer need with `scancel`.

## 4. Set a realistic time limit

Use `--time` close to what you need. Short, accurate limits help the scheduler fit your job, and everyone else's, into gaps.

## 5. Be considerate with big runs

- If you plan to take many GPUs for several days, let the lab know first.
- If you need priority before a deadline, ask the admins early.

## 6. Keep `/home` clean

There are no quotas. Everyone shares the same disks, and nothing is backed up. See [Storage](/server/storage).

## 7. Keep your account to yourself

- Don't share your account, password, or SSH key.
- Don't start services that are open to the network (e.g. Jupyter on `0.0.0.0`). Use SSH port forwarding instead, and ask the admins if you need help.

## 8. Leaving the lab

Tell the admins before you leave, and move any data you want to keep.
