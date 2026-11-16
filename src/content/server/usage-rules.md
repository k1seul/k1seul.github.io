---
title: "Shared Server Rules"
description: "Basic practices that keep shared resources available for everyone."
section: laal
language: en
order: 6
date: "2026-11-16"
---

About 40 lab members share these resources. You do not need to know every setting immediately; follow these principles and discuss unusually large runs ahead of time.

## Run experiments through Slurm

Use `master` for editing, git, and job submission. Training, evaluation, and substantial preprocessing belong in Slurm even if they do not need a GPU. See the [job examples](/server/slurm).

## Request what you need

Start with four CPUs and 30–60GB RAM per GPU, then adjust after measuring a run. Reserving excessive CPU or RAM can prevent other jobs from starting despite free GPUs.

Set a realistic runtime with some margin. Accurate limits help the scheduler fit jobs into available gaps.

## Release resources when finished

Use `exit` for interactive shells and `scancel <JOBID>` for unnecessary batch jobs. Submit when your experiment is ready; do not keep unused GPUs allocated with sleep loops.

Tell the lab and admins in advance if you need many GPUs for several days or priority near a deadline.

## Protect files and accounts

A full `/home` can cause other users' checkpoint writes to fail. Follow the [storage guide](/server/storage). Do not share accounts, passwords, or SSH private keys.

Bind personal services such as Jupyter to loopback and use SSH forwarding rather than exposing them to the network. Ask the admins if you need help.

## Before leaving the lab

Arrange data transfer and retention with the admins before graduation or departure. Keep separate copies and identify projects that need handover before your account is disabled.

