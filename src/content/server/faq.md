---
title: "FAQ & Troubleshooting"
description: "Common problems and who to ask."
order: 5
date: "2026-11-16"
---

## My process was killed / I'm out of GPU memory

Check who is using the GPU with `nvidia-smi`, and look for your own leftover processes:

```bash
nvidia-smi
ps -u $USER -f | grep python
```

## I can't connect

> **TODO:** Common causes (VPN/network, wrong port, key not registered) and what to try.

## Contact

> **TODO:** How to reach the admin, and what to include when reporting a problem (server name, time, error message).
