---
title: "GSDS Storage Server Specs"
description: "The planned server model, CPU, memory, and storage capacity at a glance."
section: gsds
language: en
order: 5
date: "2026-11-16"
---

> **Planned configuration.** The server is not in service yet; final specifications will be confirmed after purchase and installation.

LAAL plans to operate this storage server at GSDS for datasets, checkpoints, and experiment results.

| Item | Initial configuration |
| --- | --- |
| Server | Dell PowerEdge R730xd, 2U rack server |
| CPU | 2 × Intel Xeon E5-2630 v4 — 20 cores / 40 threads in total |
| Memory | 128GB DDR4 ECC |
| Storage | 4 × 18TB HDD; approximately 54TB after redundancy, before filesystem overhead and reserved free space |

The capacity above is for the whole server, not a per-user allocation. See [GSDS Storage](/server/gsds) for access and capacity requests.
