---
title: "Check Your Account & Access"
description: "After registration, verify your assigned path and permission to read and write."
section: gsds
language: en
order: 4
date: "2026-11-16"
---

> **For registered users after launch.** If you have not received a storage path, start with [Request Access](/server/gsds-access).

## 1. Check your GSDS identity

Run these in the GSDS computing environment where you will use storage:

```bash
hostname
whoami
id
```

Confirm the username matches your application. Tell the admin if your GSDS account or group membership changes.

## 2. Check your assigned path

Replace `/path/from/admin` with **your personal folder** supplied by the admin. Project folders may have different ownership and permissions.

```bash
ls -ld /path/from/admin
ls -ldn /path/from/admin
df -h /path/from/admin
```

The first command shows owner/group names; the second shows numeric UID/GID. For your personal folder, compare its owner UID with `id -u`. Shared folders may instead show a project owner or group.

Free space in `df` describes the filesystem, not necessarily your personal allocation. Use the capacity stated in your registration confirmation.

## 3. Test reading and writing

This creates a uniquely named temporary file in **your own folder**, reads it, then removes only that file:

```bash
# Enter the personal folder supplied by the admin
(
set -e
cd /path/from/admin
test_file=$(mktemp ./access-check.XXXXXX)
printf 'storage access check\n' > "$test_file"
cat "$test_file"
rm -- "$test_file"
)
```

If the text is printed and the file is removed successfully, basic read/write/delete permissions work. Do not test on another user's folder or shared datasets.

## If the path is missing or access is denied

Record the exact error, such as `Permission denied`, `No such file or directory`, or a capacity-limit error, and send it with the node and username. Do not create a replacement directory or try to fix permissions with `sudo` while access is broken.
