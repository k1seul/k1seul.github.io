---
title: "PaliGemma from Scratch"
date: "2026-05-16"
description: "Hand-written PyTorch reimplementation of the PaliGemma vision-language model (SigLIP + Gemma), from patch embeddings to KV-cached generation."
tags:
  - "VLM"
  - "Transformers"
  - "PyTorch"
external_url: "https://github.com/k1seul/paligemma"
---

**Repository:** [k1seul/paligemma](https://github.com/k1seul/paligemma)

To understand vision-language models from the inside, I reimplemented **PaliGemma** (a SigLIP vision encoder + a Gemma language decoder) entirely in PyTorch, writing every module by hand, then checked it against the reference implementation and loaded the pretrained weights.

## What's Implemented

**1. SigLIP vision encoder**

- Patch embedding with a single `Conv2d` (no `[CLS]` token) and learned position embeddings
- Multi-head attention, MLP, encoder layers, full vision model

**2. Gemma language decoder**

- RMSNorm, rotary position embeddings (RoPE)
- Grouped-query attention (GQA), GeGLU MLP
- Decoder layers and the full model

**3. Multimodal integration**

- Linear projector from image features into the language embedding space
- Input processor that merges image tokens with the text prompt
- The combined PaliGemma model

**4. Inference**

- KV cache
- Autoregressive text generation
- Loading pretrained weights

Each part comes with unit tests (`test/siglip`, `test/gemma`, `test/multimodal`).
