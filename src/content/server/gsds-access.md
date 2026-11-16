---
title: "Request Access & Register Your Account"
description: "What to send and how admins will add your storage account."
section: gsds
language: en
order: 2
date: "2026-11-16"
---

> **Applications will open after launch.** This is the planned registration process, not confirmation that an account has already been added.

## Before applying

Sign in to the **GSDS login environment** with your GSDS account and run:

```bash
hostname
whoami
id
```

Do not confuse this with output from LAAL's `master`. UID identifies your user; GID and groups identify your groups. Matching these numbers on storage keeps file ownership and shared-folder permissions consistent.

## What to send

DM Hanseul Kim with:

```text
SNU email:
GSDS username:
LAAL username, if any:
Output of id on GSDS:
Required capacity and purpose:
Shared project or members, if any:
Expected usage period:
```

Do not send passwords or SSH private keys. Share account details in a private message, not a public post.

## How your account will be added

1. An admin checks your GSDS account and lab group information.
2. The admin registers a storage user with the **same username, numeric UID/GID, and required supplementary groups as GSDS**, checking for conflicts first.
3. The admin sets ownership, permissions, and capacity limits for your personal or project folder.
4. You receive a registration confirmation, storage path, and allocated capacity.

This adds a storage user corresponding to your existing GSDS identity. It does **not** issue a new GSDS account or synchronize LAAL/GSDS passwords. Use the path from your GSDS computing environment; you do not need a separate SSH login to the storage server.

## How do I know registration is complete?

After the admin confirms registration and sends a path, follow [Check Your Account & Access](/server/gsds-operations). Do not change home directories or experiment output paths before receiving that information.

