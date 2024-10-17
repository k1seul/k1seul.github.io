---
title: "LIBERO Review"
date: 2024-10-15
permalink: /posts/2024/10/LIBERO/
tags:
  - Review
  - Reinforcement Learning
---

# LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning 논문 리뷰

#### NeurIPS 2023 (oral presentation at RAP4Robots, ICRA 2023, and TGR, CoRL 2023)

### Selection Reason:

- 표면적: Continual Reinforcement Learning 관련 주제 탐색중, 이를 Benchmarking하기 위한 Task 환경이 없을까해서 읽음

- 내면적: 2023년 논문인데 41회나 인용됨. 로봇, 멀티에이전트 시스템의 대가, [Peter Stone](<https://en.wikipedia.org/wiki/Peter_Stone_(professor)>)이 저자.

### Author Information

#### First: [Bo Liu](https://cranial-xix.github.io/)

최근 1저자 논문명

1. Longhorn: State Space Models Are Amortized Online Learners (Oral presentation at ENLSP@NeurIPS 2024)
2. Communication Efficient Distributed Training with Distributed Lion (NeurIPS 2024)
3. LLM+P: Empowering Large Language Models with Optimal Planning Proficiency

#### Last: [Peter Stone](<https://en.wikipedia.org/wiki/Peter_Stone_(professor)>)

최고 citation papers:

1. [Transfer learning for reinforcement learning domains: A survey.](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=qnwjcfAAAAAJ&citation_for_view=qnwjcfAAAAAJ:kNdYIx-mwKoC), 2009
2. [Deep recurrent q-learning for partially observable mdps](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=qnwjcfAAAAAJ&citation_for_view=qnwjcfAAAAAJ:g8uWPOAv7ggC), 2015
3. [Multiagent systems: A survey from a machine learning perspective](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=qnwjcfAAAAAJ&citation_for_view=qnwjcfAAAAAJ:u5HHmVD_uO8C), 2000

## Abstract

Generalist agent의 학습(learning)과 적응(adapting)을 위해여 Lifelong learning 패러다임을 연구해야 한다. 더 구체적으로는 강화학습과 연관지어 환경과 작용하고 행동을 선택하는 _lifelong learning in decision-making_ (**LLDM** )을 해결하는 모델을 만들어야 하는데, 이의 지식 transfer의 능력을 세분화하여 정량적으로 평가하고 확장가능한 환경이 부족하다.

따라서 이 논문은 **LIBERO** 환경을 제시한다. 저자가 제시한 이 환경에서 세분화하여 평가하는 요소는 다음과 같다.

1. How to efficiently transfer knowledge (환경 특성)
   - Declarative Knowledge (high-level의 어느정도 추상적인 정보)
   - Procedural Knowledge (low-level의 구체적인 행위에 대한 정보)
   - Mixture of both
2. How to design effective policy architectures (RNN, ViT등의 비교)
3. Effective Algorithm for LLDM (ER, EWC, PackNet)
4. The robustness of lifelong learner with respect to task ordering(환경특성)
5. The effect of model pretraining for LLDM

## Introduction

### Tasks

저자는 LLDM의 knowledge transfer를 명확한 측정을 위해, knowledge를 두 종류로 나눈 후, 한 종류만 고정하는 통제 환경 및 두 조건을 다 바꾸는 다변수 환경을 만들었다.

- Declarative Knowledge (high-level의 어느정도 추상적인 정보)
- Procedural Knowledge (low-level의 구체적인 행위에 대한 정보)

만약 후라이팬을 버너 위에 올리는 task가 있다 가정하자. 초기 후라이팬의 위치는 고정 되었지 않다면,
"후라이팬을 버너 위에 올려야 된다"는 high level의 Declarative Knowledge은 변하지 않지만,
초기 후라이팬 위치에 따라 움직여야 하는 low level의 근육의 Procedural Knowledge은 변한다.

이러한 지식의 종류를 변수로 설정하여 LIBERO는 모델의 부족한 지식 전달 방법을 정확하게 평가할 수 있다.

#### Environments

- LIBERO-Object (Different layouts, same objects) Declarative Knowledge transfer 평가
- LIBERO-Spatial (Different object, same layout) Procedural Knowledge transfer 평가
- LIBERO-Goal (Different goals, same objects & layout) 둘다 transfer 평가
- LIBERO-100 (Diverse objects, layouts, background) (YOLO)

\*\* 잘 생각해보면 Declarative Knowledge transfer 평가는 새로운 Procedural Knowledge에 대한 adaptability를 평가하는 것이랑 같다.

## Background and Architecture

### Math

MDP is defined as $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, H, \mu_0, R)$ s.t

- $\mathcal{S}, \mathcal{A}$ are the state and action spaces of the robot
- $\mu_0$ is the initial state distribution (시작 물건 위치가 랜덤성이 있기 때문)
- $\mathcal{R} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ is the reward function
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ is the transition function
  In this work, assume
- sparse reward setting
- replace $R$ with goal predicate $g : \mathcal(S) \rightarrow \{0, 1\}$
  so the expected return is
  $J(\pi) = \mathbb{E}_{s_t, a_t ~ \pi, \mu_0}[\sum_{t=1}^H(g(s_t)]$

The _lifelong robot learning problem_ can be defined using

- $K$ tasks $\{T^1, \dots, T^k\}$
- policy $\pi$ is conditioned on the task $\pi (\cdot | s;T)$
- $T^k = \mu_0^k, g^k)$
  The robot aims to optimize
  $\max_{\pi} J_{LRL}(\pi) = \frac{1}{k}\sum_{p=1}^k\bigg[ \mathbb{E}_{s_t^p, a_t^p ~ \pi(c\dot ; T^p), \mu_0^p }\big[\sum_{t=1}^L g^p(s_t^p)\big] \bigg]$

Also, this paper used _Imitation Learning_ for inducing search space similar to the demonstration

- Denote $D^k = \{ \tau_i^k\}_{i=1}^N$ as $N$ demonstrations for task $T^k$
- $\tau_i^k = (o_0, a_0, o_1, a_1, \dots, o_{l^k})$ where $l^k \leq H$
- To induce MDP from non-Markovian env, $s_t$ is represented by aggregated history of observations,  
  $s_y \equiv o_{\leq t} = (o_0, o_1, \dots, o_t)$
- Behavior cloning loss is
  $min_\pi J_{BC}(\pi) = \frac{1}{k}\sum^k_{p=1} \mathbb{E}_{o_t, a_t ~ D^p} \bigg[ \sum_{t=0}^{l^p} \mathcal{L}(\pi(o_{\leq t}; T^p), a_t^p) \bigg]$

### Non-math

LIBERO 는 Procedurally generation으로 다음과 같이 동작함.

1. [Ego4D](https://ego4d-data.org/) 데이터 셋을 이용한 템플렛 및 language annotations 추출
2. [PPDL](https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language) 언어를 통한 구체화된 시작 분포 $\mu_0$ 형성
3. goal 명시화

- _unary predicates_ (한 물건의 상태 e.g. Open($X$), TurnOff($X$))
- _binary predicates_ (물건 간의 상태 e.g. On($A$, $B$)

## Experiment

### Searching Space

#### Algorithms

- Experience Replay(ER)
- Elastic Weight Consolidation (EWC)
- PackNet (SOTA)

#### Neural Network Architectures

- ResNet-RNN
- ResNet-T (Transformer decoder)
- Vit-T

### Evaluation Metric

- Forward Transfer (FWT), higher better
  $FWT = \sum_{k \in [K]} \frac{FWT_k}{K}, FWT_k = \frac{1}{11} \sum_{e \in \{ 0 \dots 50\} c_{k,k,e}}$
  average success rates of current task $k$ across multiple epochs

- Negative backward transfer (NBT), lower better
  $NBT = \sum_{k \in [K]} \frac{NBT_k}{K}, NBT_k = \frac{1}{K-k}\sum^K_{\tau=k+1}(c_{k,k}-c_{\tau, k})$
  average of future agents best score across training epochs

- AUC (combination), higher better
  $AUC = \sum_{k \in [K]}\frac{AUC_k}{K}, AUC_k = \frac{1}{K-k+1}(FWT_k + \sum^K_{\tau = k+1}c_{\tau, k})$

## Findings

### Q1, Q2

**Q1**: How do different architectures/LL algorithms perform under specific distribution shifts
**Q2** To what extent does neural architecture impact knowledge transfer in LLDM, and are there any discernible patterns in the specialized capabilities of each architecture?
ResNet-T and Vit-T work much better than ResNet-RNN on average.

> Written with [StackEdit](https://stackedit.io/).
