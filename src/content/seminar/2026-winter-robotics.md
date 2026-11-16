---
title: "2026 Winter: Robotics Study"
date: "2026-12-01"
period: "From December 2026"
status: "planning"
institution: "LAAL, Seoul National University"
image: "/images/seminar/2026-winter-robotics.svg"
description: "The second term moves past foundational VLAs to three threads: advanced VLAs, world action models, and LLM agents as robot policies. It opens with a NeurIPS Day."
---

The winter term builds on the summer [VLA study](/seminar/2026-summer-vla). Since the foundational VLA papers are now shared background, this term follows three threads that are moving fast right now: **advanced VLAs**, **world action models**, and **LLMs and agents as robot policies**. The schedule is still being planned; this page will fill in as sessions are assigned.

## Plan

- **Kick-off: NeurIPS Day** (first week of December, date to be decided). A 3–4 hour session to read through NeurIPS 2026 papers quickly and share them: everyone brings the papers they find interesting and introduces the core idea.
- **Eight weekly sessions** after the kick-off, drawn from the candidate list below.
- **One main talk plus two 5-minute intros.** Each session has one main presenter who goes deep on a paper, and two participants who each give a 5-minute introduction to another paper. This covers more papers per week and keeps everyone presenting.
- **Same ground rules as the summer.** Everyone presents, slides go to the shared folder before the session, and sessions are in person with an online option.

## Themes

| Theme | Question |
| --- | --- |
| Advanced VLA | How do VLAs gain memory, learn from their own experience, and scale with human data? |
| World Action Models (WAM) | What do we gain by predicting future observations and actions together, and by using world models to improve policies? |
| LLM as Policy | Can general-purpose LLM agents act as robot policies by writing, calling, and revising code and tools? |

The summer's Core track is gone, since those papers are now shared background. Each candidate is tagged **Background**, **Advanced**, or **Frontier**, with importance and difficulty ratings as before.

## Candidate papers

### Advanced VLA

| Paper | Main idea | Track | Importance | Difficulty |
| --- | --- | --- | --- | --- |
| [π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://www.pi.website/download/pi07.pdf) (Technical Report 2026) | Multimodal prompts enable compositional robot generalization. | Frontier | ★★★★★ | ★★★★☆ |
| [MolmoAct2: Action Reasoning Models for Real-world Deployment](https://allenai.org/papers/molmoact2) (Technical Report 2026) | Embodied reasoning VLM with a flow-matching action expert. | Frontier | ★★★★★ | ★★★☆☆ |
| [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/abs/2603.03596) (arXiv 2026) | Short-term video memory with long-term semantic text memory. | Frontier | ★★★★★ | ★★★★☆ |
| [NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation](https://arxiv.org/abs/2607.06678) (CoRL 2026) | Compresses each past frame-view into one native memory token. | Frontier | ★★★★★ | ★★★★☆ |
| [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/abs/2604.23073) (arXiv 2026) | Compact VLA readout enables lightweight online actor-critic RL. | Frontier | ★★★★★ | ★★★★☆ |
| [EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data](https://arxiv.org/abs/2602.16710) (arXiv 2026) | Large-scale egocentric human data transfers to dexterous robot skills. | Frontier | ★★★★★ | ★★★★☆ |
| [ODE Policy: Robot Learning via Action Dynamics, Not Actions](https://arxiv.org/abs/2611.01729) (arXiv 2026) | Learns action dynamics and integrates continuous robot motion. | Frontier | ★★★★☆ | ★★★★☆ |
| [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)](https://arxiv.org/abs/2502.19645) (RSS 2025) | Parallel action chunks and simple L1 regression improve VLA fine-tuning. | Advanced | ★★★★☆ | ★★★☆☆ |
| [π\*0.6: a VLA That Learns From Experience (RECAP)](https://arxiv.org/abs/2511.14759) (arXiv 2025) | Combines demonstrations, rollouts and human corrections through RL. | Advanced | ★★★★☆ | ★★★★☆ |
| [X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model](https://arxiv.org/abs/2510.10274) (ICLR 2026) | Soft prompts adapt flow policies to heterogeneous robot embodiments. | Advanced | ★★★★☆ | ★★★☆☆ |

### World Action Models

| Paper | Main idea | Track | Importance | Difficulty |
| --- | --- | --- | --- | --- |
| [World Action Models are Zero-shot Policies (DreamZero)](https://arxiv.org/abs/2602.15922) (arXiv 2026) | Joint future-video and action prediction from a video backbone. | Frontier | ★★★★★ | ★★★★☆ |
| [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning](https://arxiv.org/abs/2601.16163) (ICLR 2026) | Video diffusion predicts actions, future states and values. | Frontier | ★★★★★ | ★★★★☆ |
| [VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model](https://arxiv.org/abs/2602.10098) (ECCV 2026) | Learns action-relevant transitions through future latent prediction. | Frontier | ★★★★★ | ★★★★☆ |
| [UNITAS: A 3D-Native World Action Model for Embodied Manipulation](https://arxiv.org/abs/2610.12099) (arXiv 2026) | Unifies action and scene dynamics in a shared metric 3D space. | Frontier | ★★★★★ | ★★★★☆ |
| [World Action Planner: Generalizable Robot Decision-Making with Action-Conditioned World Models](https://arxiv.org/abs/2607.27599) (arXiv 2026) | Refines action plans through world-model imagination and search. | Frontier | ★★★★★ | ★★★★☆ |
| [Agentic World Action Models (AW-0)](https://arxiv.org/abs/2610.30136) (arXiv 2026) | Jointly models executable code, future video and robot actions. | Frontier | ★★★★★ | ★★★★☆ |
| [LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning](https://arxiv.org/abs/2611.14048) (arXiv 2026) | Interaction-centric latent reasoning links world prediction and action. | Frontier | ★★★★☆ | ★★★★☆ |
| [World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy](https://arxiv.org/abs/2602.06508) (arXiv 2026) | Co-improves a video world model and VLA through virtual policy RL. | Advanced | ★★★★☆ | ★★★★☆ |
| [World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry](https://arxiv.org/abs/2604.01985) (NeurIPS 2026) | Verifies state plausibility and action reachability to improve models. | Advanced | ★★★★☆ | ★★★★☆ |
| [WMPO: World Model-based Policy Optimization for Vision-Language-Action Models](https://arxiv.org/abs/2511.09515) (arXiv 2025) | Optimizes a VLA with on-policy RL inside a pixel-based world model. | Advanced | ★★★★☆ | ★★★★☆ |
| [WorldVLA: Towards Autoregressive Action World Model](https://arxiv.org/abs/2506.21539) (arXiv 2025) | Autoregressively models images and actions in one framework. | Advanced | ★★★★☆ | ★★★☆☆ |
| [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) (arXiv 2025) | Self-supervised video latents support action-conditioned planning. | Advanced | ★★★★☆ | ★★★★☆ |

### LLM as Policy

| Paper | Main idea | Track | Importance | Difficulty |
| --- | --- | --- | --- | --- |
| [Agent as Policy for Robotic Manipulation (AGP)](https://arxiv.org/abs/2609.12541) (arXiv 2026) | A general-purpose agent observes, codes, acts and revises from feedback. | Frontier | ★★★★★ | ★★★☆☆ |
| [Make Code as Policy Great Again: Frontier Agents Write, Call, and Evolve Robot Tools (URAI)](https://arxiv.org/abs/2609.39018) (arXiv 2026) | Agents build, call and improve reusable robot tools from feedback. | Frontier | ★★★★★ | ★★★☆☆ |
| [Robo-Harness K1: Harnessing Robot-Use Agents via Perception Augmentation](https://arxiv.org/abs/2609.29389) (arXiv 2026) | Exposes calibrated perception as tools for robot-use agents. | Frontier | ★★★★★ | ★★★☆☆ |
| [RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks](https://arxiv.org/abs/2603.11558) (arXiv 2026) | One VLM orchestrates collection, policy learning and long-horizon use. | Frontier | ★★★★★ | ★★★☆☆ |
| [OpenRUA: Robot-Use Agents Are Zero-Shot Visuomotor Policies](https://arxiv.org/abs/2610.02459) (arXiv 2026) | Coding agents use terminal and ROS 2 to construct visuomotor policies. | Frontier | ★★★★☆ | ★★★☆☆ |
| [Self-Evolving Coding Agents: From Digital Programs to Physical-World Intelligence (Physical Coding)](https://arxiv.org/abs/2609.35432) (arXiv 2026) | Represents world state and policy as editable, verifiable programs. | Frontier | ★★★★☆ | ★★★☆☆ |
| [ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation](https://arxiv.org/abs/2409.01652) (arXiv 2024) | VLM-generated keypoint constraints are optimized into robot motions. | Background | ★★★☆☆ | ★★★☆☆ |
| [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753) (ICRA 2023) | LLMs generate executable policy programs calling robot APIs. | Background | ★★★☆☆ | ★★☆☆☆ |
