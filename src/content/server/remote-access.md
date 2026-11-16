---
title: "Off-Campus Access (Cloudflare WARP)"
description: "Install Cloudflare WARP to reach the servers from outside SNU."
section: laal
order: 2
date: "2026-11-16"
---

Outside the SNU network, the LAAL servers are only reachable through **Cloudflare WARP**. Install it first, then ask the admin how to connect.

## Install the WARP client

| Platform | How to install |
| --- | --- |
| Windows / macOS | Download **Cloudflare WARP** from [one.one.one.one](https://one.one.one.one/) and install it. |
| Ubuntu / Debian | Use the commands below. |
| iOS / Android | Install **Cloudflare One Agent** from the app store. |

On Ubuntu / Debian:

```bash
curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg \
  | sudo gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/cloudflare-client.list
sudo apt update && sudo apt install cloudflare-warp
```

## Next step

After installing, **DM Hanseul Kim**. The admin will send you the login and connection details.
