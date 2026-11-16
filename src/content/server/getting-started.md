---
title: "처음 시작하기"
description: "계정 신청부터 SSH 접속, 첫 GPU 작업까지 순서대로 따라가세요."
section: laal
order: 1
date: "2026-11-16"
---

처음 오셨다면 **계정 신청 → 서버 접속 → GPU 할당 → 실험 실행** 순서로 진행하면 됩니다. 이 안내는 LAAL 서버 기준입니다. GSDS 계정과 스토리지는 [GSDS 안내](/server/gsds)에서 별도로 확인해 주세요.

## 1. 계정 신청

김한슬에게 DM으로 **학교 이메일과 원하는 사용자 이름**을 보내 주세요. 사용자 이름은 본인을 알아볼 수 있는 영문 소문자와 숫자로 정하면 됩니다. 계정이 준비되면 접속 주소와 초기 비밀번호를 전달드립니다.

이 문서의 `<username>`, `<server-address>`는 본인의 계정과 안내받은 주소로 바꾸세요. 꺾쇠도 지웁니다. 서버 주소·포트·비밀번호는 공개 문서에 올리지 않습니다.

## 2. 내 컴퓨터에서 접속

교외에서는 먼저 [Cloudflare WARP 연결](/server/remote-access)을 마쳐 주세요. macOS/Linux는 터미널, Windows는 PowerShell에서 실행합니다.

```bash
# 내 컴퓨터에서 실행
ssh <username>@<server-address>
```

별도 SSH 포트를 안내받았다면 `ssh -p <port> <username>@<server-address>`를 사용합니다. 첫 연결 시 호스트 키 지문을 관리자에게 받은 값과 비교한 뒤 승인하세요.

접속 후 아래 명령으로 확인합니다.

```bash
# 접속한 서버에서 실행
hostname
whoami
yppasswd
```

`hostname`이 로그인 노드 `master`인지, `whoami`가 내 계정인지 확인합니다. `yppasswd`는 초기 비밀번호를 바꾸는 명령입니다. LAAL은 NIS로 계정을 공유하므로 여기서는 `passwd` 대신 사용합니다.

## 3. 첫 GPU 작업 실행

로그인 노드는 코드 편집과 작업 제출용입니다. GPU가 필요한 실험은 Slurm이 계산 노드를 할당한 뒤 실행합니다.

<figure>
  <a href="/images/server/laal-workflow.svg" aria-label="LAAL 작업 흐름 그림 크게 보기">
    <img src="/images/server/laal-workflow.svg" width="800" height="430" alt="내 컴퓨터에서 master로 SSH 접속한 뒤 Slurm으로 GPU 노드를 할당받습니다. 모든 LAAL 노드는 master의 공유 home을 사용하고, 로컬 data는 해당 노드만 접근합니다." />
  </a>
  <figcaption>접속과 자원 할당은 별도 단계입니다. GPU 작업이 끝나면 자원을 반납합니다. 그림을 누르면 크게 볼 수 있습니다.</figcaption>
</figure>

```bash
# master에서 실행: GPU 1개를 30분 동안 요청
launch-shell 1 30 laal_3090 16
```

자원이 비어 있지 않으면 대기할 수 있습니다. 할당 후 프롬프트가 바뀌면 아래를 실행하세요.

```bash
# 할당받은 계산 노드에서 실행
hostname
nvidia-smi
exit
```

노드 이름과 GPU가 출력되면 첫 작업은 성공입니다. `exit`로 쉘을 닫으면 자원을 반납하고 `master`로 돌아옵니다. Python 환경은 [실험 환경 만들기](/server/environment), 오래 실행할 실험은 [Slurm 작업 실행](/server/slurm)을 참고하세요.

## 4. 매번 비밀번호를 입력하지 않으려면

내 컴퓨터에 SSH 키가 없다면 생성합니다. 이미 사용하는 키가 있으면 새로 만들거나 덮어쓰지 않아도 됩니다. 생성 시 암호(passphrase) 설정을 권장합니다.

```bash
# 내 컴퓨터에서 실행
ssh-keygen -t ed25519 -C "your_name@laptop"
ssh-copy-id <username>@<server-address>
```

별도 포트는 `ssh-copy-id -p <port> ...`로 지정합니다. `ssh-copy-id`가 없는 Windows 등에서는 공개키 파일 `~/.ssh/id_ed25519.pub`의 내용을 복사한 뒤, 서버에서 다음과 같이 등록합니다.

```bash
# 서버에서 실행
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

편집기에서 기존 키는 남겨 두고 새 공개키를 한 줄로 추가합니다. nano에서는 Ctrl+O, Enter로 저장하고 Ctrl+X로 나옵니다. **공개키(.pub)만 복사하고 개인키는 공유하지 마세요.** 기존 연결을 열어 둔 채 새 터미널에서 접속을 시험합니다.

## 5. VS Code와 짧은 접속 이름

내 컴퓨터의 `~/.ssh/config`에 아래를 추가하면 `ssh laal`로 접속할 수 있습니다. 별도 포트라면 `Port <port>`도 추가하세요.

```ssh-config
Host laal
    HostName <server-address>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

VS Code에서는 **Remote - SSH** 확장을 설치하고 `Remote-SSH: Connect to Host...`에서 `laal`을 선택합니다. 이때 통합 터미널도 처음에는 `master`에 열리므로, 학습 전에 Slurm으로 자원을 요청해야 합니다.

다음으로 [파일 저장 위치](/server/storage)와 [함께 쓰는 규칙](/server/usage-rules)을 확인하면 준비가 끝납니다. 막히는 부분은 [문제 해결](/server/faq)에서 먼저 찾아보세요.

