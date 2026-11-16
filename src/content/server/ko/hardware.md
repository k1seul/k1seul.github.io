---
title: "서버와 GPU 선택하기"
description: "로그인 노드와 GPU 노드의 역할, 파티션 선택 방법을 안내합니다."
section: laal
language: ko
order: 3
date: "2026-11-16"
---

`master`는 SSH로 접속해 코드를 편집하고 작업을 제출하는 로그인 노드입니다. `laal1`~`laal3`는 Slurm으로 할당받아 실험하는 계산 노드입니다. `/home`은 노드 간 공유되지만, 각 노드의 로컬 디스크는 공유되지 않습니다.

## 현재 운영 중인 장비

| 노드 | GPU | CPU | RAM | Slurm 파티션 |
| --- | --- | --- | --- | --- |
| `master` | 없음 | Xeon Silver 4510, 12코어 | 512GB | 로그인 전용 |
| `laal1` | RTX 3090 24GB × 8 | Xeon Gold 6326 × 2, 64스레드 | 512GB | `laal_3090` |
| `laal2` | RTX A6000 48GB × 10 | Xeon Silver 4410Y × 2, 48스레드 | 768GB | `laal_a6000` |
| `laal3` | RTX PRO 6000 Blackwell Max-Q 96GB × 4 | EPYC 9654, 192스레드 | 640GB | `laal_rtx6000` |

이 표는 관리자가 기록한 구성을 정리한 것으로, 실시간 가용 자원은 아래 명령을 기준으로 확인하세요. 현재 Slurm의 CPU 단위는 하드웨어 스레드입니다.

## 어떤 GPU를 고르면 되나요?

먼저 모델과 batch가 필요로 하는 **GPU 한 장의 메모리**를 기준으로 선택합니다.

| 필요한 VRAM | 후보 |
| --- | --- |
| 24GB 이내 | `laal_3090` |
| 24GB 초과, 48GB 이내 | `laal_a6000` |
| 48GB 초과, 96GB 이내 | `laal_rtx6000` |

여러 GPU를 요청해도 GPU 메모리가 자동으로 합쳐지지 않습니다. 다중 GPU 학습은 코드에서 지원해야 합니다. Blackwell은 [소프트웨어 호환성](/ko/server/environment)도 확인해 주세요.

실험이 두 GPU 종류 모두에서 동작한다면 배치 스크립트에 여러 파티션을 지정할 수 있습니다.

```bash
#SBATCH --partition=laal_a6000,laal_3090
```

Slurm이 우선순위와 가용 자원을 고려해 배치합니다. 특정 노드에 직접 SSH하여 GPU를 점유하지 말고, [Slurm 안내](/ko/server/slurm)를 따라 주세요.

## 현재 사용량 확인

```bash
# master에서 실행
sinfo
squeue -u "$USER"
sinfo -N -O "NodeList:10,Partition:14,Gres:22,GresUsed:28,CPUsState:16,StateLong:10"
```

`GresUsed`는 할당된 GPU, `CPUsState`는 할당/유휴/기타/전체 CPU 수입니다. 실제 GPU 사용률과 Slurm 할당량은 다를 수 있습니다.

## 증설 예정 장비

LAAL 4는 MSI CG480-S6053 4U 플랫폼으로 도입을 검토하고 있습니다. **초기 GPU 1장, 향후 최대 8장** 확장 계획이며 아직 위 운영 장비 목록에는 포함하지 않습니다. 최종 RAM·GPU 구성과 파티션은 도입 후 공지합니다.

GSDS 스토리지 이용과 계정 등록은 [별도 안내](/ko/server/gsds-overview)를 참고하세요.
