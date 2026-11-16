---
title: "GSDS Storage Overview"
description: "Service status, eligibility, and the steps from requesting access to using your space."
section: gsds
language: en
order: 1
date: "2026-11-16"
---

> **Not open yet.** The lab will announce the launch date and storage paths once confirmed.

## What is this service?

Additional storage for LAAL members running experiments at GSDS. It is intended for datasets, checkpoints, and results used across GSDS compute nodes.

It does not automatically replace LAAL's existing `/home` or your GSDS account.

**LAAL operates the storage and handles all storage incidents.** GSDS mainly assists with mounting the storage on its compute nodes. Contact a LAAL admin for storage access or failures.

## Who can use it?

The service is planned for LAAL members with a GSDS account and access to the LAAL group (`lab01`). If you do not have a GSDS account, follow the GSDS account application process first.

## How do I get started?

<figure>
  <a href="/images/server/gsds-storage-plan.svg" aria-label="Open the storage registration flow">
    <img src="/images/server/gsds-storage-plan.svg" width="800" height="410" alt="Check your GSDS account, apply to a LAAL admin, have your storage account registered with matching user and group IDs, then verify your assigned path and access." />
  </a>
  <figcaption>This is the planned process after launch. The admin will send your personal storage path once registration is complete.</figcaption>
</figure>

1. Send your GSDS account information and required capacity using the [application guide](/server/gsds-access).
2. A LAAL admin registers your storage user and access permissions.
3. Once you receive your path, [check your account and access](/server/gsds-operations).

## Where should I store files until launch?

Keep using existing GSDS space at `/shared/s2/lab01/$USER`. The launch notice will explain when requests open, where to store files, and your allocated capacity.

See [Policies & Support](/server/gsds-management) for capacity requests, backups, and help.

