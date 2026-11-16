---
title: "GSDS 스토리지 이용 안내"
description: "서비스 상태, 이용 대상과 신청부터 사용 시작까지의 흐름을 안내합니다."
section: gsds
language: ko
order: 1
date: "2026-11-16"
---

> **아직 개통 전입니다.** 서비스 시작일과 저장 경로가 확정되면 연구실에 공지합니다.

## 어떤 서비스인가요?

GSDS에서 실험하는 LAAL 구성원을 위한 추가 저장 공간입니다. 학습 데이터셋, 체크포인트와 실험 결과를 보관하고 여러 계산 노드에서 사용할 수 있도록 준비하고 있습니다.

LAAL 서버의 기존 `/home`이나 GSDS 계정을 자동으로 대체하는 서비스는 아닙니다.

**스토리지 운영과 장애 대응은 LAAL이 전담합니다.** GSDS는 주로 계산 노드에 저장소를 연결하는 마운트 작업을 도와줍니다. 저장소 접근이나 장애 문제는 LAAL 관리자에게 문의해 주세요.

## 누가 이용할 수 있나요?

GSDS 계정과 LAAL 그룹(`lab01`) 이용 권한이 있는 LAAL 구성원을 대상으로 합니다. GSDS 계정이 없다면 먼저 GSDS의 계정 발급 절차를 따라 주세요.

## 어떻게 이용을 시작하나요?

<figure>
  <a href="/images/server/gsds-storage-plan.ko.svg" aria-label="스토리지 계정 신청 흐름 크게 보기">
    <img src="/images/server/gsds-storage-plan.ko.svg" width="800" height="410" alt="GSDS 계정 확인 후 LAAL 관리자에게 신청합니다. 관리자가 동일한 사용자·그룹 번호로 스토리지 사용자를 등록하고, 사용자가 안내받은 경로와 읽기·쓰기 권한을 확인합니다." />
  </a>
  <figcaption>개통 후 적용할 이용 절차입니다. 신청과 등록이 완료되면 관리자가 개인별 저장 경로를 안내합니다.</figcaption>
</figure>

1. [신청 방법](/ko/server/gsds-access)에 따라 GSDS 계정 정보와 필요한 용량을 전달합니다.
2. LAAL 관리자가 스토리지 사용자와 접근 권한을 등록합니다.
3. 경로를 안내받으면 [계정·권한 확인](/ko/server/gsds-operations)을 진행합니다.

## 개통 전에는 어디에 저장하나요?

기존 GSDS 공간인 `/shared/s2/lab01/$USER`를 계속 사용해 주세요. 새 공간의 신청 시작일, 저장 경로와 개인별 사용 가능 용량은 개통 공지에서 안내합니다.

백업·용량·문의 방법은 [이용 정책과 문의](/ko/server/gsds-management)를 확인해 주세요.
