---
title: "실험 환경 만들기"
description: "개인 Python 환경을 만들고, 계산 노드에서 GPU 사용을 확인합니다."
section: laal
language: ko
order: 7
date: "2026-11-16"
---

패키지는 개인 환경에 설치합니다. 시스템 Python이나 NVIDIA 드라이버를 직접 변경할 필요는 없습니다. 아래 두 방법 중 **conda 또는 uv 하나**를 선택하세요.

## conda 사용

이미 `conda`가 있으면 재설치하지 마세요. 없다면 [Miniconda 공식 안내](https://www.anaconda.com/docs/getting-started/miniconda/install)에 따라 내 홈에 설치합니다. 기본 설치 경로는 `~/miniconda3`입니다.

```bash
# master에서 실행
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n myenv python=3.11
conda activate myenv
```

프로젝트의 `requirements.txt`나 `environment.yml`이 있으면 그 버전을 우선합니다. PyTorch 설치 명령은 [공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 Linux / Pip / 사용할 CUDA 조합을 확인하세요.

배치 스크립트에서는 새 쉘이 열리므로 매번 환경을 활성화해야 합니다.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myenv
python -u train.py
```

## uv 사용

`uv`가 없다면 [공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 개인 계정에 설치합니다.

```bash
# master의 프로젝트 폴더에서 실행
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

배치 스크립트도 같은 프로젝트 폴더에서 `source .venv/bin/activate` 후 실행합니다. 프로젝트 파일이 없다면 필요한 패키지를 개별 설치하세요.

## GPU가 보이는지 확인

`master`에는 GPU가 없으므로 GPU 검사는 할당받은 계산 노드에서 합니다.

```bash
# master에서 자원 요청
launch-shell 1 30 laal_3090 16
# 계산 노드 쉘이 열린 후 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
exit
```

마지막 출력이 `True`이면 해당 환경에서 GPU를 사용할 수 있습니다. conda 예시이므로 uv를 선택했다면 활성화 줄만 바꾸세요.

## CUDA와 Blackwell 호환성

기존 운영 기록은 아래와 같습니다. 드라이버와 toolkit은 점검 후 바뀔 수 있으므로 계산 노드에서 `nvidia-smi`, 필요하면 `nvcc --version`으로 확인합니다.

| 항목 | laal1 | laal2 | laal3 |
| --- | --- | --- | --- |
| OS | Ubuntu 22.04 | Ubuntu 22.04 | Ubuntu 22.04 |
| NVIDIA 드라이버 계열 | 580 | 580 | 580 |
| /usr/local CUDA 개발 도구 | 12.1, 12.2, 12.8, 13.0 | 11.7, 12.2, 13.0 | 12.8, 13.0 |

`nvidia-smi`의 CUDA 표시는 드라이버가 지원하는 수준이며, Python 환경에 설치된 CUDA 실행 라이브러리 버전과 다를 수 있습니다. PyTorch 배포 패키지은 실행 라이브러리을 포함하는 경우가 많아, 단순 실행에 시스템 개발 도구이 반드시 필요한 것은 아닙니다.

`laal3`의 Blackwell은 해당 GPU를 지원하는 빌드가 필요합니다. 오래된 CUDA·PyTorch 환경을 그대로 가져오면 실패할 수 있으니, [PyTorch 공식 버전 안내](https://pytorch.org/get-started/previous-versions/)에서 Blackwell 지원 조합을 선택하세요. 같은 `/home`을 쓰더라도 GPU 종류와 확장 모듈 호환성까지 같아지는 것은 아닙니다.

CUDA 확장 모듈을 직접 컴파일할 때만, 실제 노드에 존재하고 프로젝트와 맞는 개발 도구을 지정합니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

## 기타 도구

일반 사용자에게 Docker 권한은 제공하지 않습니다. conda·uv를 사용하고 다른 실행 환경이 필요하면 관리자에게 문의하세요. AI 코딩 CLI 등 개인 개발 도구는 내 홈에 설치할 수 있습니다. 로그인 노드에서 코드 편집은 가능하지만, 도구가 실행하는 학습·평가도 Slurm 안에서 진행해야 합니다.

환경과 캐시도 공유 용량을 사용합니다. [스토리지 안내](/ko/server/storage)를 참고해 사용하지 않는 환경을 정리해 주세요.
