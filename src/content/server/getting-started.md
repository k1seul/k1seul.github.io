---
title: "Getting Started"
description: "Request an account, set up an SSH key, and log in for the first time."
order: 1
date: "2026-11-16"
---

## 1. Request an account

> **TODO:** How to request an account (who to contact and what to send: preferred username, SSH public key, expected usage period).

## 2. Create an SSH key (on your laptop)

```bash
ssh-keygen -t ed25519 -C "your_name@laptop"
cat ~/.ssh/id_ed25519.pub   # send this PUBLIC key to the admin
```

Never share the private key (`~/.ssh/id_ed25519`).

## 3. Add the server to your SSH config

Use the address, port, and username you received from the admin. Add this to `~/.ssh/config` on your laptop:

```ssh-config
Host laal
    HostName <server-address>
    Port <port>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
```

Then connect with:

```bash
ssh laal
```

VS Code users can connect the same way with the **Remote - SSH** extension. Just pick `laal` from the host list.

## 4. First login checklist

> **TODO:** First-login steps, e.g. change the initial password, where to put your code and data, and read the [Usage Rules](/server/usage-rules).
