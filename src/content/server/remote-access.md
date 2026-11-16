---
title: "Off-Campus Access"
description: "Install Cloudflare WARP, enroll in the lab organization, and check SSH access."
section: laal
language: en
order: 2
date: "2026-11-16"
---

Off campus, install **Cloudflare One Client (WARP)** on your own computer and enroll in the lab organization. Turning on a personal WARP connection alone does not grant lab access.

## 1. Install the client

Use the stable release for your OS from [Cloudflare's official download page](https://developers.cloudflare.com/cloudflare-one/team-and-resources/devices/cloudflare-one-client/download/). Linux users should follow the distribution-specific instructions there. Install it on **the computer you connect from**, not on the lab server.

## 2. Enroll in the lab organization

DM Hanseul Kim with your SNU email and operating system for the organization name and login instructions.

1. Open the client's organization/Zero Trust login option. Labels may vary by version.
2. Enter the organization name you receive and complete browser authentication.
3. Confirm the organization and connection status in the client.

Check that you are using the lab organization rather than another institution's or personal WARP. See [Cloudflare's manual enrollment guide](https://developers.cloudflare.com/cloudflare-one/team-and-resources/devices/cloudflare-one-client/deployment/manual-deployment/) for screen-by-screen help.

## 3. Test SSH

```bash
# On your computer
ssh <username>@<server-address>
```

Use the username, address, and any nonstandard port sent by the admin. Then continue with [Getting Started](/server/getting-started).

## If it does not connect

| Symptom | Check first |
| --- | --- |
| Connection times out | Organization enrollment, WARP status, server address and port |
| Permission denied | Username, password or registered SSH key |
| Connection drops | Other VPNs, client status, maintenance notices |

Send your OS, time, error message, and client connection status to the admin if needed. Do not send passwords or private keys. This setup is for LAAL; GSDS access follows GSDS instructions.

