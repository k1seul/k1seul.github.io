---
title: "문제가 생겼을 때"
description: "SSH 접속, Slurm 대기, 메모리 부족과 저장 오류를 확인하는 순서입니다."
section: laal
order: 8
date: "2026-11-16"
---

먼저 에러 원문과 작업 ID를 남겨 주세요. 아래 확인으로 해결되지 않으면 [관리자에게 보낼 정보](/server/administration#reporting-a-problem)를 함께 전달하면 됩니다.

## 서버에 접속되지 않아요

| 증상 | 확인 순서 |
| --- | --- |
| 연결 시간 초과 | 교외라면 WARP 조직·연결 확인 → 주소·포트 확인 → 점검 공지 확인 |
| `Permission denied` | 계정 이름 → 비밀번호 또는 SSH 키 등록 확인 |
| 호스트 키 변경 경고 | 연결을 멈추고 관리자에게 지문 확인. 경고를 지우고 바로 접속하지 않기 |

[교외 접속 안내](/server/remote-access)와 [SSH 키 설정](/server/getting-started)을 참고하세요.

## 작업이 계속 대기 중이에요

```bash
# master에서 실행
squeue -u "$USER"
sinfo
```

Reason이 `Resources`라면 GPU뿐 아니라 요청한 CPU·RAM도 함께 비어야 합니다. `Priority`는 스케줄러 우선순위 대기입니다. 대기 이유별 설명은 [Slurm 안내](/server/slurm)에 있습니다.

## 학습이 종료됐어요

```bash
sacct -j <JOBID> -o JobID,State,ExitCode,Elapsed,MaxRSS
```

| 상태 또는 로그 | 의미 | 다음 행동 |
| --- | --- | --- |
| `OUT_OF_MEMORY` | CPU RAM 요청량 초과 가능성 | 수집된 메모리 사용량과 데이터 로더를 확인하고 `--mem` 조정 |
| `TIMEOUT` | 실행 시간 한도 도달 | checkpoint 재개 지원과 `--time` 확인 |
| `CUDA out of memory` | GPU VRAM 부족 | batch 축소, 혼합 정밀도 또는 더 큰 VRAM의 GPU 검토 |
| 시작 직후 `FAILED` | 환경·경로·코드 문제 등 | 해당 작업 로그의 첫 에러 확인 |

`MaxRSS`가 비어 있거나 일부 단계만 표시될 수 있습니다. 로그도 함께 확인해 주세요. CPU RAM과 GPU VRAM은 다른 자원입니다.

## GPU가 보이지 않거나 너무 느려요

먼저 `hostname`으로 GPU가 없는 `master`에서 실행 중인지 확인합니다. Slurm 할당 안이라면 `nvidia-smi`와 Python 환경을 확인하세요. Blackwell에서 `no kernel image is available`가 보이면 [환경 호환성](/server/environment)을 살펴봅니다.

GPU 사용률이 낮다고 항상 GPU 자체의 문제는 아닙니다. 데이터 읽기·전처리·CPU 사용·다중 GPU 동기화 등을 확인하고, 공유 HDD가 병목이면 [로컬 scratch](/server/storage) 사용을 검토하세요.

## No space left on device가 나와요

에러가 난 **실제 출력 폴더**를 대상으로 확인합니다.

```bash
df -h .
df -i .
du -sh .
```

바이트 용량, inode, quota 또는 임시 디렉터리 문제일 수 있습니다. 본인의 불필요한 파일만 정리하고 공유 파일은 임의로 지우지 마세요. 여러 사용자에게 발생하면 관리자에게 알려 주세요.

## 파일을 실수로 지웠어요

자동 백업은 없습니다. 추가 쓰기와 정리를 멈추고 관리자에게 바로 알려 주세요. 다른 위치의 사본을 확인할 수는 있지만 복구를 보장하지는 못합니다.

## GSDS 신규 스토리지가 안 보여요

현재 준비 단계라 아직 마운트가 제공되지 않습니다. [진행 상태](/server/gsds-overview)를 확인하고, 개통 공지 전에는 기존 GSDS 공간을 사용해 주세요.

