---
title: "Storage Policies & Support"
description: "Capacity requests, shared folders, backups, maintenance, and who to contact."
section: gsds
language: en
order: 3
date: "2026-11-16"
---

## Who operates the storage?

| Team | Responsibility |
| --- | --- |
| LAAL | Setup, accounts, permissions, capacity, monitoring, maintenance, and all storage incident investigation, repair, and user notices |
| GSDS | Mainly mounting the storage on GSDS compute nodes |

GSDS does not manage or troubleshoot the lab storage for us. **Report all storage failures to LAAL**, including a missing or inaccessible storage path. LAAL investigates and coordinates any required mount changes with GSDS.

## Capacity and shared data

The default personal allocation is not finalized. It will be announced at launch. Request extra space with its purpose and expected duration. For datasets used by several people, request a project folder to avoid duplicate copies.

Reaching your capacity limit can cause new writes, including checkpoints, to fail. Ask for changes before errors occur. Remove your own temporary files and unnecessary checkpoints when experiments end.

## Backups and retention

Automatic backups are not provided. Keep code and irreplaceable final results in another location too. Do not modify or delete other users' shared data.

Before graduation, departure, or the end of a project, arrange retention and data transfer with the admin.

## Maintenance and interruptions

Storage may be unavailable during maintenance or incidents. Planned maintenance will be announced in the lab. If jobs stall or writes fail, record the job ID and error rather than repeatedly retrying.

## Who should I contact?

| Admin | Requests |
| --- | --- |
| Hanseul Kim | Registration, permissions, capacity changes, storage errors |
| Minsoo Jo | Urgent incidents when the main admin is unavailable |

DM the time, GSDS node, job ID, storage path, and exact error. Do not publish internal addresses or account details. GSDS account issuance and group membership follow GSDS procedures; **contact LAAL first for lab storage access problems**.

