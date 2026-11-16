---
title: "내 계정과 권한 확인"
description: "등록 완료 안내를 받은 뒤 저장 경로와 내 폴더의 읽기·쓰기 권한을 확인합니다."
section: gsds
language: ko
order: 4
date: "2026-11-16"
---

> **개통 후 등록 완료 안내를 받은 사용자용입니다.** 안내받은 저장 경로가 아직 없다면 먼저 [이용 신청](/ko/server/gsds-access)을 확인해 주세요.

## 1. GSDS 계정 확인

스토리지를 사용할 GSDS 계산 환경에서 실행합니다.

```bash
hostname
whoami
id
```

신청한 사용자 이름과 일치하는지 확인합니다. GSDS 계정이나 그룹이 변경되면 관리자에게도 알려 주세요.

## 2. 내 저장 경로 확인

아래 `/path/from/admin`을 관리자가 안내한 **내 개인 폴더** 경로로 바꿉니다. 프로젝트 공유 폴더는 별도로 안내받은 권한을 따릅니다.

```bash
ls -ld /path/from/admin
ls -ldn /path/from/admin
df -h /path/from/admin
```

첫 명령은 소유자·그룹 이름, 두 번째는 숫자 UID/GID를 보여 줍니다. 내 개인 폴더의 사용자 번호가 `id -u`와 맞는지 확인하세요. 공유 폴더는 프로젝트 소유자나 그룹으로 표시될 수 있습니다.

`df`의 여유 용량은 전체 저장 공간 기준이며, 나에게 배정된 사용 가능 용량과 같지는 않습니다. 개인별 용량은 관리자의 등록 안내를 기준으로 확인합니다.

## 3. 읽기·쓰기 시험

내 폴더에 다른 파일과 겹치지 않는 임시 파일을 만들고, 읽은 뒤 그 파일만 지우는 예시입니다.

```bash
# 안내받은 내 개인 폴더로 이동합니다.
(
set -e
cd /path/from/admin
test_file=$(mktemp ./access-check.XXXXXX)
printf 'storage access check\n' > "$test_file"
cat "$test_file"
rm -- "$test_file"
)
```

문구가 출력되고 임시 파일 삭제까지 완료되면 기본적인 읽기·쓰기·삭제 권한이 있는 것입니다. 다른 사용자의 폴더나 공유 데이터로 시험하지 마세요.

## 경로가 없거나 권한 오류가 나면

`Permission denied`, `No such file or directory`, 용량 제한 오류 등을 그대로 기록하고, 사용한 노드·계정과 함께 관리자에게 보내 주세요. 연결 문제가 있는 상태에서 같은 이름의 폴더를 임의로 만들거나, `sudo`·권한 변경 명령으로 해결하려고 하지 않습니다.
