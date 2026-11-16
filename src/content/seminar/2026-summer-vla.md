---
title: "2026 Summer: VLA Study"
date: "2026-07-03"
period: "July – September 2026"
status: "completed"
institution: "LAAL, Seoul National University"
image: "/images/seminar/2026-summer-vla.jpg"
description: "Eleven weekly sessions on vision-language-action models, from CLIP and RT-1 to π0 and π0.5: 22 papers, two per session, with slides and recordings."
sessions:
  - week: 0
    date: "Jul 3, 2026"
    track: "Background"
    image: "/images/seminar/2026-summer-vla-orientation.jpg"
    papers:
      - title: "Learning Transferable Visual Models From Natural Language Supervision (CLIP)"
        venue: "ICML 2021"
        url: "https://arxiv.org/abs/2103.00020"
        idea: "Contrastive pretraining on image–text pairs gives transferable visual features and zero-shot classification."
      - title: "PaliGemma: A versatile 3B VLM for transfer"
        venue: "Google DeepMind 2024"
        url: "https://arxiv.org/abs/2407.07726"
        idea: "A compact SigLIP vision encoder and Gemma language model combine into an efficient, high-performing vision-language model."
        slides: "https://docs.google.com/presentation/d/1MWtV8e8OUOk3w4sbA5aXdcS6ub6oZIYkk0M2SvbUQt8/edit"
        video: "https://youtu.be/Ygz0-Ku5pP8"
  - week: 1
    date: "Jul 10, 2026"
    track: "Background"
    topic: "Action Modeling"
    image: "/images/seminar/2026-summer-vla-w1-rl.jpg"
    papers:
      - title: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
        venue: "NeurIPS 2021"
        url: "https://arxiv.org/abs/2106.01345"
        idea: "Recasts reinforcement learning as conditional sequence modeling: a causal Transformer predicts actions from desired returns, past states, and actions."
        slides: "https://drive.google.com/file/d/1M3mu1gqKeaEOSiMXyKJbu-BNA2ioAgDZ/view"
        video: "https://youtu.be/swmKkfkfRjw"
      - title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
        venue: "RSS 2023"
        url: "https://arxiv.org/abs/2303.04137"
        idea: "Models visuomotor control as conditional denoising diffusion, so the policy can learn multimodal, high-dimensional action distributions stably."
        slides: "https://drive.google.com/file/d/10qiubwXTvdmHpaSem2IzG3WMmaAuv21a/view"
        video: "https://youtu.be/swmKkfkfRjw"
  - week: 2
    date: "Jul 21, 2026"
    track: "Background"
    image: "/images/seminar/2026-summer-vla-w2-palm-e.jpg"
    papers:
      - title: "PaLM-E: An Embodied Multimodal Language Model"
        venue: "ICML 2023"
        url: "https://arxiv.org/abs/2303.03378"
        idea: "Feeds visual, textual, and continuous sensor inputs directly into a large language model for embodied reasoning and robot planning."
        slides: "https://drive.google.com/file/d/1Tey9_wcjJr1X_gOo4ngjC0vbnqpV75tI/view"
        video: "https://youtu.be/b6f6KA0vX7I"
      - title: "RT-1: Robotics Transformer for Real-World Control at Scale"
        venue: "RSS 2023"
        url: "https://arxiv.org/abs/2212.06817"
        idea: "A Transformer policy trained on large, diverse real-world robot data follows language instructions and generalizes to new tasks and environments."
        slides: "https://docs.google.com/presentation/d/1IucdyjTG8HMPaYLbdmVmzGgst3zOv0VsNmVv1xVI_r0/edit"
        video: "https://youtu.be/c4Vqq8pm2Zk"
  - week: 3
    date: "Jul 28, 2026"
    track: "Core"
    topic: "Foundational VLA"
    image: "/images/seminar/2026-summer-vla.jpg"
    papers:
      - title: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"
        venue: "CoRL 2023"
        url: "https://arxiv.org/abs/2307.15818"
        idea: "Introduces the VLA paradigm: co-fine-tunes vision-language models on web data and robot trajectories, with actions represented as tokens."
        slides: "https://drive.google.com/file/d/1sF2ZX5pbg6IyoN4wlku2G0KCl_uCR6iQ/view"
        video: "https://youtu.be/b6f6KA0vX7I"
      - title: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
        venue: "CoRL 2023"
        url: "https://arxiv.org/abs/2310.08864"
        idea: "Aggregates large-scale data from many robot embodiments and shows that cross-robot training (RT-X) transfers positively across platforms and tasks."
        slides: "https://docs.google.com/presentation/d/1xzqART-_sW4TfACjzLI_KDOvvA_f0uao/edit"
        video: "https://youtu.be/b6f6KA0vX7I"
  - week: 4
    date: "Aug 4, 2026"
    track: "Core"
    topic: "Foundational VLA"
    image: "/images/seminar/2026-summer-vla-w4-octo.jpg"
    papers:
      - title: "Octo: An Open-Source Generalist Robot Policy"
        venue: "RSS 2024"
        url: "https://arxiv.org/abs/2405.12213"
        idea: "An open-source Transformer policy trained on cross-embodiment data, conditioned on language or goal images, and quick to adapt to new robots and action spaces."
        slides: "https://drive.google.com/file/d/1Y4edqnoe8LyrPSEQUygeHjvM0R0WE1yJ/view"
        video: "https://youtu.be/vVk7p1RRX_I"
      - title: "OpenVLA: An Open-Source Vision-Language-Action Model"
        venue: "CoRL 2024"
        url: "https://arxiv.org/abs/2406.09246"
        idea: "A 7B open-source VLA built from pretrained vision and language backbones, with strong generalist manipulation and efficient fine-tuning."
        slides: "https://drive.google.com/file/d/17-6jW4gGFJnwCSPTKkM1sisY7dKlDdfm/view"
        video: "https://youtu.be/vVk7p1RRX_I"
  - week: 5
    date: "Aug 11, 2026"
    track: "Core"
    topic: "Action Modeling"
    image: "/images/seminar/2026-summer-vla-w5-pi0.jpg"
    papers:
      - title: "π0: A Vision-Language-Action Flow Model for General Robot Control"
        venue: "arXiv 2024"
        url: "https://arxiv.org/abs/2410.24164"
        idea: "Pairs a pretrained vision-language model with a flow-matching action expert to produce continuous, high-frequency actions for dexterous manipulation."
        slides: "https://drive.google.com/file/d/1zrrK9bNwH8F8WrGFfNQsgJzsDXmPbbqg/view"
        video: "https://youtu.be/iGaeT-_PVr8"
      - title: "FAST: Efficient Action Tokenization for Vision-Language-Action Models"
        venue: "ICML 2025"
        url: "https://arxiv.org/abs/2501.09747"
        idea: "Tokenizes action sequences in frequency space with the discrete cosine transform, making high-frequency actions practical for autoregressive VLAs."
        slides: "https://docs.google.com/presentation/d/1wlLYpmcjgUs-b7OG3K3a3cIkzvqUo0KHZtx1We8s_es/edit"
        video: "https://youtu.be/iGaeT-_PVr8"
  - week: 6
    date: "Aug 18, 2026"
    track: "Core"
    topic: "Data Scaling"
    image: "/images/seminar/2026-summer-vla.jpg"
    papers:
      - title: "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"
        venue: "CoRL 2024"
        url: "https://arxiv.org/abs/2403.12945"
        idea: "A large real-world manipulation dataset collected across many scenes, tasks, and locations to improve policy performance and generalization."
        slides: "https://drive.google.com/file/d/1OrmTYIpRehv1xyzRg0ERQyRioQKs9mYY/view"
        video: "https://youtu.be/XntAjJ-01WE"
      - title: "BridgeData V2: A Dataset for Robot Learning at Scale"
        venue: "CoRL 2023"
        url: "https://arxiv.org/abs/2308.12952"
        idea: "A large, diverse manipulation dataset across many environments and tasks, built for scalable multi-task and language-conditioned learning."
        slides: "https://drive.google.com/file/d/1OrmTYIpRehv1xyzRg0ERQyRioQKs9mYY/view"
        video: "https://youtu.be/XntAjJ-01WE"
  - week: 7
    date: "Aug 25, 2026"
    track: "Advanced"
    topic: "Reasoning VLA"
    image: "/images/seminar/2026-summer-vla-w7-cot-vla.jpg"
    papers:
      - title: "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models"
        venue: "CVPR 2025"
        url: "https://arxiv.org/abs/2503.22020"
        idea: "Generates explicit visual chain-of-thought steps (future subgoal images) so the robot reasons about what to do before acting."
        slides: "https://docs.google.com/presentation/d/1SuVNh3EyLKUEspW-7jY8ugp8tIKh9ayI/edit"
        video: "https://youtu.be/-qPAIQIb3wo"
      - title: "ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model"
        venue: "ICLR 2025"
        url: "https://arxiv.org/abs/2502.14420"
        idea: "Unifies multimodal conversation and robot control in one model that understands images and language while producing actions."
        slides: "https://docs.google.com/presentation/d/1SuVNh3EyLKUEspW-7jY8ugp8tIKh9ayI/edit"
        video: "https://youtu.be/-qPAIQIb3wo"
  - week: 8
    date: "Sep 1, 2026"
    track: "Advanced"
    topic: "Safety / World Model & Planning"
    papers:
      - title: "SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning"
        venue: "NeurIPS 2025 Spotlight"
        url: "https://arxiv.org/abs/2503.03480"
        idea: "Aligns VLA policies with explicit safety constraints through constrained reinforcement learning."
      - title: "WMPO: World Model-based Policy Optimization for Vision-Language-Action Models"
        venue: "ICLR 2026"
        url: "https://arxiv.org/abs/2511.09515"
        idea: "Improves a VLA with on-policy reinforcement learning inside a pixel-based world model instead of the real robot."
  - week: 9
    date: "Sep 8, 2026"
    track: "Advanced"
    topic: "Efficient VLA / World Model & Planning"
    papers:
      - title: "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation"
        venue: "IEEE RA-L 2025"
        url: "https://arxiv.org/abs/2409.12514"
        idea: "Builds compact VLAs on small multimodal backbones with a diffusion-policy head for fast inference and data-efficient training."
      - title: "WorldVLA: Towards Autoregressive Action World Model"
        venue: "arXiv 2025"
        url: "https://arxiv.org/abs/2506.21539"
        idea: "Autoregressively models images and actions in one framework, combining a VLA with an action-conditioned world model."
  - week: 10
    date: "Sep 15, 2026"
    track: "Frontier"
    topic: "Foundational VLA"
    papers:
      - title: "π0.5: a Vision-Language-Action Model with Open-World Generalization"
        venue: "arXiv 2025"
        url: "https://arxiv.org/abs/2504.16054"
        idea: "Co-trains on heterogeneous robot, web, and high-level semantic data so the policy generalizes to entirely new homes."
      - title: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"
        venue: "arXiv 2025"
        url: "https://arxiv.org/abs/2503.14734"
        idea: "A dual-system humanoid VLA: a vision-language module for reasoning and a diffusion-Transformer module for motor actions, trained on real, human-video, and synthetic data."
---

The first term of the LAAL Robotics Seminar was a summer reading study on **vision-language-action models (VLAs)**. Over eleven weekly sessions we read 22 papers, two per session, from the vision-language and policy-learning background through foundational VLAs to recent reasoning, safety, and world-model work.

## Curriculum

I started from a candidate list of 42 papers and sorted each one into a **track** and a **research topic**. The schedule moved through the tracks in order, so each session could build on the ones before it.

| Track | What it covers |
| --- | --- |
| Background | Prerequisites in vision, language, robotics, and policy learning that VLAs build on |
| Core | Must-read papers that define the key concepts and architectures of VLAs |
| Advanced | Papers that go deep into a specific VLA research topic |
| Frontier | Recent papers at the edge of VLA research |

The research topics, each framed as a question:

| Topic | Key question |
| --- | --- |
| Foundational VLA | How do we build a generalist VLA policy? |
| Action Modeling | How can robot actions be represented and generated more effectively? |
| Data Scaling | How can robot data be scaled efficiently to improve VLAs? |
| Embodiment | How can VLA policies generalize across robot embodiments? |
| Reasoning VLA | Can reasoning improve the generalization of VLA policies? |
| World Model & Planning | Can world models and planning enable long-horizon, goal-directed manipulation? |
| Post-training / Adaptation | How can pretrained VLAs be adapted efficiently to new tasks and environments? |
| Efficient / On-device | How can VLAs run efficiently on resource-constrained robots? |
| Safety | How can robot behavior stay safe and reliable in open-world environments? |

Each entry in the reading list also carried a one-line main idea and two star ratings, **importance** and **difficulty**, so presenters could pick papers that suited them. A few frontier papers were kept in reserve to swap in after week 8: [Qwen-VLA](https://arxiv.org/abs/2605.30280), [MolmoAct2](https://arxiv.org/abs/2605.02881), [VLA-JEPA](https://arxiv.org/abs/2602.10098), [Cosmos 3](https://arxiv.org/abs/2606.02800), [Gemini Robotics](https://arxiv.org/abs/2503.20020), and [Evaluating Gemini Robotics Policies in a Veo World Simulator](https://arxiv.org/abs/2512.10675).

## How it ran

- **Every participant presents at least twice.** Papers were assigned from each person's preferred order, and adjusted when choices collided.
- **Slides go up before the session.** Presenters upload their slides (PDF or PPT) to the shared study folder before the session starts.
- **In person first, online when needed.** Sessions were held offline at the lab, with an online option for anyone who could not attend.
- **Every session is recorded.** I recorded the sessions and uploaded them to YouTube, so the slides and recordings below form an archive of the study.

## My role

I organized the study end to end: the track and topic structure, the reading list, the schedule and presenter assignments, the orientation session, and the recordings. I also presented PaliGemma (week 0), RT-1 (week 2), and FAST (week 5), and was scheduled for GR00T N1 (week 10).
