---
title: "Slurm으로 GPU 작업 실행하기"
description: "대화형 작업, 배치 스크립트, 로그 확인과 자원 반납을 예제로 안내합니다."
section: laal
language: ko
order: 4
date: "2026-11-16"
---

Slurm은 여러 사람이 GPU·CPU·메모리를 나누어 쓰도록 작업을 배치하는 도구입니다. 아래의 제출·조회 명령은 별도 설명이 없으면 **로그인 노드 master**에서 실행합니다.

| 하려는 일 | 방법 |
| --- | --- |
| 코드 확인, 디버깅, 짧은 테스트 | `launch-shell`로 대화형 쉘 열기 |
| 터미널을 닫아도 계속될 학습 | `sbatch`로 배치 작업 제출 |

## 대화형 작업

```bash
launch-shell <GPU개수> [시간_분] [파티션] [메모리_GB]
```

| 인자 | 생략 시 값 | 참고 |
| --- | --- | --- |
| GPU 개수 | 필수 | 0~4개 |
| 시간 | 60분 | 최대 480분 |
| 파티션 | `laal_3090` | [GPU 종류 확인](/ko/server/hardware) |
| CPU 메모리 | GPU당 10GB, GPU 0개면 10GB | GPU 메모리와 별개 |

CPU는 `2 × GPU개수 + 2`개가 할당됩니다. 예를 들어 A6000 한 장, 2시간, CPU 메모리 40GB는 다음과 같습니다.

```bash
launch-shell 1 120 laal_a6000 40
# 할당이 끝나고 계산 노드 쉘이 열리면 실행
nvidia-smi
# 준비한 Python 환경을 활성화한 뒤 실험 실행
python train.py
exit
```

마지막 `exit`가 자원을 반납합니다. 시간이 끝나면 작업도 종료되므로, 긴 실험은 아래 배치 방식을 사용하세요.

## 배치 작업

실험 폴더에서 `run.sh`를 만들고 아래 내용을 저장합니다. 먼저 [Python 환경](/ko/server/environment)을 만들고 `myenv`를 실제 환경 이름으로 바꾸세요.

```bash
#!/bin/bash
#SBATCH --job-name=my_exp
#SBATCH --partition=laal_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myenv
python -u train.py
```

이 예시는 Miniconda 기본 설치 경로 기준입니다. uv 환경이라면 `source .venv/bin/activate`로 두 환경 활성화 줄을 대신합니다. `#SBATCH` 줄은 쉘 명령보다 위에 두고, 경로에는 `$USER` 같은 쉘 변수를 넣지 마세요.

```bash
# master의 실험 폴더에서 실행
mkdir -p slurm_logs
sbatch run.sh
```

`Submitted batch job 405`처럼 나오면 접수된 것입니다. 실행 시작 여부는 별도로 확인합니다. 로그 폴더는 Slurm이 만들지 않으므로 **제출 전에** 생성해야 합니다. 기본 작업 폴더는 제출 명령을 실행한 위치입니다.

## 상태, 로그, 취소

```bash
squeue -u "$USER"
tail -f slurm_logs/my_exp-405.out
scancel 405
```

`405`는 실제 작업 ID로 바꾸세요. 로그를 보는 `tail -f`는 Ctrl+C로 나와도 학습이 계속됩니다. 작업 자체를 끝내려면 `scancel`을 사용합니다. `scancel -u "$USER"`는 **내 작업 전체**를 취소하므로 필요할 때만 사용하세요.

실행 중인 작업의 GPU를 확인하려면 기존 할당 안에서 도구를 실행할 수 있습니다.

```bash
srun --jobid=405 --overlap --pty nvtop
# nvtop이 없으면
srun --jobid=405 --overlap nvidia-smi
```

완료 또는 실패한 작업은 다음으로 확인합니다. 메모리 측정값은 클러스터의 수집 설정에 따라 비어 있을 수 있으며, MaxRSS는 작업 단계별 최대 프로세스 값입니다.

```bash
sacct -j 405 -o JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

## 자원은 얼마나 요청하나요?

| 자원 | 시작점 | 조정 방법 |
| --- | --- | --- |
| CPU | GPU당 4개 | 데이터 로더 사용량에 맞춰 조정 |
| CPU 메모리 | GPU당 30~60GB | 첫 실행의 실측 사용량을 확인 |
| GPU | 실제 사용할 개수 | 자동으로 여러 GPU를 쓰지는 않음 |
| 시간 | 예상 실행 시간 + 여유 | 한도 초과 시 작업이 종료됨 |

`--mem`은 **노드의 CPU RAM**입니다. GPU VRAM을 늘리는 옵션이 아닙니다. 노드당 요청 가능한 메모리 상한은 현재 `laal1`·`laal3` 500G, `laal2` 566G입니다. 실제 설정은 `scontrol show partition`과 관리자 안내를 기준으로 확인하세요.

GPU 사용률이 낮으면 데이터 읽기, 전처리, CPU, 동기화 등을 함께 살펴보세요. CPU를 늘리는 것만으로 해결되지 않을 수 있습니다.

## 대기 중일 때

| squeue의 Reason | 의미와 다음 행동 |
| --- | --- |
| `Resources` | 필요한 자원이 함께 비기를 기다립니다. CPU·RAM을 과하게 요청하지 않았는지 확인하세요. |
| `Priority` | 스케줄러 우선순위가 높은 작업이 먼저입니다. 반드시 제출 순서는 아닙니다. |
| `Dependency` | 선행 작업의 완료 조건을 기다립니다. |
| `ReqNodeNotAvail` | 필요한 노드가 사용 불가 상태입니다. 공지나 관리자에게 확인하세요. |

`sinfo`로 다른 파티션 상태를 확인할 수 있습니다. 시간 한도나 QOS 등 다른 이유가 표시되면 작업 ID와 함께 관리자에게 문의하세요.

## 여러 실험과 후속 작업

아래는 먼저 제출한 405번 작업이 성공하면 평가를 실행하는 예입니다.

```bash
sbatch --dependency=afterok:405 test.sh
```

파라미터 탐색은 작업 배열로 관리할 수 있습니다. 아래 줄을 배치 스크립트에 넣고, 실험 코드가 배열 인덱스를 읽도록 작성하세요. `%4`는 동시 실행을 4개로 제한합니다.

```bash
#SBATCH --array=0-99%4
# 환경 활성화 등 기존 설정 다음에 실행
python sweep.py --idx "$SLURM_ARRAY_TASK_ID"
```

더 자세한 옵션은 [Slurm 공식 sbatch 문서](https://slurm.schedmd.com/sbatch.html)에서 확인할 수 있습니다.
