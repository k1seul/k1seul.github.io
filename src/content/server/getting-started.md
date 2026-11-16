---
title: "Getting Started"
description: "Follow account setup, SSH login, your first GPU job, and SSH key setup."
section: laal
language: en
order: 1
date: "2026-11-16"
---

New to LAAL? Follow **request an account → connect → request a GPU → run your experiment**. This guide is for LAAL servers. GSDS accounts and storage have [separate instructions](/server/gsds).

## 1. Request an account

DM or email **[Hanseul Kim](mailto:k1seul@snu.ac.kr)** with your SNU email and preferred username. Use lowercase letters and digits, and choose a name others can recognize. The admin will send your server address and initial password.

Replace `<username>`, `<server-address>`, and `<port>` below with the values you receive, removing the angle brackets. Addresses, ports, and passwords are not published here.

## 2. Connect from your computer

Off campus, complete [Cloudflare WARP setup](/server/remote-access) first. Open Terminal on macOS/Linux or PowerShell on Windows.

```bash
# On your own computer
ssh <username>@<server-address>
```

For a nonstandard port, use `ssh -p <port> <username>@<server-address>`. On first connection, compare the host key fingerprint with the admin's information before accepting.

```bash
# After connecting to the server
hostname
whoami
yppasswd
```

Check that the host is `master` and the username is yours. Change your initial password with `yppasswd`: LAAL shares accounts through NIS, so use this rather than `passwd`.

## 3. Run your first GPU job

The login node is for editing and submitting jobs. Slurm allocates a compute node before you run GPU work.

<figure>
  <a href="/images/server/laal-workflow.svg" aria-label="Open the LAAL workflow diagram">
    <img src="/images/server/laal-workflow.svg" width="800" height="430" alt="SSH connects your computer to master. Slurm allocates a GPU node. All LAAL nodes share home, while local data belongs only to its node." />
  </a>
  <figcaption>Connecting and allocating resources are separate steps. Release resources when finished. Select the figure to enlarge it.</figcaption>
</figure>

```bash
# On master: request one GPU for 30 minutes
launch-shell 1 30 laal_3090 16
```

You may wait for resources. Once the compute-node shell opens:

```bash
# Inside the allocated shell
hostname
nvidia-smi
exit
```

Seeing a compute-node name and GPU information confirms the allocation. `exit` releases it and returns you to `master`. Next, [create a Python environment](/server/environment) and use [batch jobs](/server/slurm) for longer experiments.

## 4. Set up an SSH key

If you already have an SSH key, reuse it rather than overwriting it. Otherwise create one on your computer and choose a passphrase.

```bash
# On your own computer
ssh-keygen -t ed25519 -C "your_name@laptop"
ssh-copy-id <username>@<server-address>
```

Use `ssh-copy-id -p <port> ...` for a nonstandard port. If `ssh-copy-id` is unavailable, copy the contents of your public key `~/.ssh/id_ed25519.pub` and register it on the server:

```bash
# On the server
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Keep existing keys and add yours on a new single line. In nano, save with Ctrl+O, Enter, then exit with Ctrl+X. **Only the public key (.pub) leaves your computer. Never share your private key.** Keep the existing session open while testing a new login.

## 5. Use VS Code or a shorter SSH command

Add this to `~/.ssh/config` on your computer. Add `Port <port>` if needed.

```ssh-config
Host laal
    HostName <server-address>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

Now use `ssh laal`. In VS Code, install **Remote - SSH**, run `Remote-SSH: Connect to Host...`, and choose `laal`. Its terminal starts on `master`; request a Slurm allocation before training.

Read [Storage](/server/storage) and [Usage Rules](/server/usage-rules) before a large run. For help, see [Troubleshooting](/server/faq).

