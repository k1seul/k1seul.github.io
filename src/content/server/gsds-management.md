---
title: "Storage Policies & Support"
description: "Capacity requests, shared folders, backups, maintenance, and who to contact."
section: gsds
language: en
order: 3
date: "2026-11-16"
---

## Operation and support

LAAL manages the storage, including account registration, access permissions, maintenance, and troubleshooting. If you need help or run into a problem, please contact a LAAL admin using the details below.

## Account registration and access scope

For research data access, only approved members' UID/GID values and the groups required for their access will be registered and authorized. Other user or group IDs will not be imported automatically or granted access. System and administrator accounts needed to operate the server are managed separately.

Access will also be restricted through folder permissions and access settings; creating or omitting a local account alone does not enforce these restrictions. Sharing a GSDS server or group does not automatically grant access to lab storage.

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

DM the time, GSDS node, job ID, storage path, and error message so we can look into it. Please keep internal addresses and account details in private messages.

