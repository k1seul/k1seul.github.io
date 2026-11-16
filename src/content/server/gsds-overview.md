---
title: "LAAL Storage at GSDS (Planned)"
description: "Why LAAL is adding its own storage server in the GSDS server room, how it will work, and where things stand."
section: gsds
order: 1
date: "2026-11-16"
---

> **Status: planned.** The hardware, capacity, and launch date are not fixed yet. This page will be updated as plans firm up.

## What it is

LAAL plans to buy its own storage server and install it in a rack in the **GSDS server room**. It is meant for LAAL members who run jobs on the GSDS cluster, for example with `rjob` from the LAAL servers.

## Why

- **Space next to the GSDS GPUs** for datasets, checkpoints, and results.
- **Managed by LAAL**: LAAL sets the quotas and handles requests, instead of relying only on the shared GSDS lab space.

## How it will work (planned)

- **Same account as GSDS.** Your storage account will use the same username and UID as your GSDS account, so your files keep the right owner on GSDS nodes. That means you need a GSDS account first.
- **Mounted on GSDS nodes.** The mount path will be announced at launch.
- **Per-user quotas**, with more space on request. See [Requesting Space & Quotas](/server/gsds-access).
- **Not a backup.** Like the LAAL servers, keep a copy of anything you can't afford to lose.

## Progress

| Step | Status |
| --- | --- |
| Confirm rack space in the GSDS server room | In discussion with GSDS admins |
| Get account (UID) information to match GSDS accounts | In discussion with GSDS admins |
| Choose hardware and capacity | Not decided |
| Purchase and install | Not started |
| Open for requests | After installation |

## Until it's ready

Keep using the existing GSDS lab space (`/shared/s2/lab01/$USER`) for jobs on GSDS.
