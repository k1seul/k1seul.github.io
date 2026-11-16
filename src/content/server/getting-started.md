---
title: "Getting Started"
description: "Request an account, log in for the first time, and set up SSH keys."
section: laal
order: 1
date: "2026-11-16"
---

## 1. Request an account

Send a **direct message (DM) to Hanseul Kim** with:

- your **SNU email** address
- the **username** you want. Pick one that tells others who you are, e.g. `hanseulkim` (lowercase letters and digits).

Accounts are for LAAL members. The admin will send you your initial password once the account is ready.

## 2. Log in for the first time

Use the server address the admin sends you with your account.

- **Off campus:** set up Cloudflare WARP first. See [Off-Campus Access](/server/remote-access).

```bash
ssh <username>@<server-address>
```

Change your initial password right away. Accounts are shared across all LAAL servers (NIS), so use `yppasswd`, not `passwd`:

```bash
yppasswd
```

## 3. Set up an SSH key (no more passwords)

On **your laptop**:

```bash
ssh-keygen -t ed25519 -C "your_name@laptop"    # press Enter to accept the defaults
ssh-copy-id <username>@<server-address>         # asks for your password one last time
```

If `ssh-copy-id` is not available (e.g. on Windows), add the key by hand. Print it on your laptop with `cat ~/.ssh/id_ed25519.pub`, then on the server:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys      # paste the public key on one line, save
chmod 600 ~/.ssh/authorized_keys
```

Only the **public** key (`.pub`) ever leaves your laptop. Never share the private key.

## 4. Shorter SSH command

Add this to `~/.ssh/config` on your laptop:

```ssh-config
Host laal
    HostName <server-address>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

Now `ssh laal` is enough. VS Code users can pick `laal` in the **Remote - SSH** extension.

## 5. Where you are

You land on **`master`, the login node**. It has no GPUs and is shared by everyone, so it is only for editing code, git, and submitting jobs. All experiments run on the GPU nodes through Slurm.

Next steps:

1. [Servers & Hardware](/server/hardware): what each node has
2. [Running Jobs with Slurm](/server/slurm): how to get a GPU
3. [Storage](/server/storage) and [Usage Rules](/server/usage-rules): please read before your first big run
