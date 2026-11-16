---
title: "What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction"
author: "Jewon Yeom*, Hanseul Kim*, Jeongjae Park, Sungmok Jung, Jaejin Lee, Taesup Kim (*equal contribution)"
date: "2026-06-05"
journal: "Conference on Robot Learning (CoRL 2026)"
external_url: "https://arxiv.org/abs/2606.07687"
image: "/images/publications/corl2026-action-relevant-latents.png"
description: "High pixel fidelity does not imply action-relevant features. A probe-based study across image SSL, video pretraining, autoencoders, and diffusion models shows that temporal prediction, not reconstruction quality, makes world-model latents useful for action."
tags:
  - "World Models"
  - "Robot Learning"
  - "Representation Learning"
---

**[Paper (arXiv)](https://arxiv.org/abs/2606.07687)** · **[Code](https://github.com/k1seul/corl2026-action-relevant-latents)**

## Summary

Video world models are increasingly used as predictive visual representations, but it is unclear which pretraining signals produce action-relevant structure in their latent spaces. We study this with a unified probe-based evaluation across many encoder families: image-only self-supervision, video pretraining, autoencoders, diffusion models, and dynamics models.

## Key Findings

- **The sharpest models recover no action.** The highest-PSNR models reach probe $R^2 \le 0$, while video-pretrained encoders lead in action recovery.
- **Predicting the correct action makes latents action-relevant.** Fine-tuning with an inverse-dynamics objective lifts V-JEPA 2 from $R^2 = 0.40$ to $0.84$; shuffled action labels give no gain.
- **Policies improve when objects move, not overall.** Future-feature supervision for $\pi_{0.5}$ raises LIBERO-10 success under object position shift from 8% to 24%.
