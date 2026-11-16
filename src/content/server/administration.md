---
title: "Administration"
description: "Who runs the servers, how to reach them, and how maintenance works."
section: laal
order: 9
date: "2026-11-16"
---

## Admins

| | Role | Responsible for |
| --- | --- | --- |
| **Hanseul Kim** | Main admin | accounts, Slurm and system configuration, incident response, announcements |
| **Minsoo Jo** | Deputy admin | covers when the main admin is away, hardware checks, first point of contact for user questions |

## How to reach us

- **Accounts and anything else:** DM **Hanseul Kim**.
- **Urgent and no reply:** DM **Minsoo Jo**.

The admins are students too, so non-urgent requests may take a day or two.

## Reporting a problem

Include these details and we can usually fix things much faster:

- **Node / partition**, e.g. `laal2` / `laal_a6000`
- **Job ID**, if it's about a job
- **When** it happened
- **What you ran**: the command or `sbatch` script
- **The error message**, or the path to the log file

## Maintenance

- **Regular maintenance:** in the **first week of each month**, announced in the lab channel ahead of time. Jobs may need to stop during the window.
- **Urgent maintenance:** a hardware fault, such as a GPU that stops responding, may need a reboot on short notice. We announce it as soon as we can.

## Monitoring and usage data

To keep the servers healthy and plan upgrades, the admins record usage statistics:

- GPU, CPU, and memory utilization per job
- power draw
- storage usage

They are used only to run and improve the servers.

## Accounts

- **New account:** see [Getting Started](/server/getting-started#1-request-an-account).
- **Leaving the lab:** tell the admins in advance and move any data you want to keep. Accounts of members who have left may be disabled.

## Suggestions

If something in this guide is wrong or unclear, DM Hanseul Kim.
