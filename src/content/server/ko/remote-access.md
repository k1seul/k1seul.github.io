---
title: "교외에서 접속하기"
description: "Cloudflare 클라이언트 설치, 연구실 조직 등록, SSH 연결 확인 순서입니다."
section: laal
language: ko
order: 2
date: "2026-11-16"
---

교외에서 LAAL 서버에 접속하려면 내 컴퓨터에 **Cloudflare One Client(WARP)**를 설치하고 연구실에서 안내하는 조직에 등록해야 합니다. 앱을 켜는 것만으로 연구실 접근 권한이 생기지는 않습니다.

## 1. 클라이언트 설치

[Cloudflare 공식 다운로드](https://developers.cloudflare.com/cloudflare-one/team-and-resources/devices/cloudflare-one-client/download/)에서 내 OS용 안정 버전을 설치합니다. Linux도 공식 페이지의 배포판별 설치 방법을 사용해 주세요. 이 설치는 서버가 아니라 **접속에 사용할 내 컴퓨터**에서 진행합니다.

## 2. 연구실 조직에 등록

김한슬에게 학교 이메일과 사용하는 OS를 DM으로 보내면 조직 이름과 로그인 절차를 안내드립니다.

1. 클라이언트에서 조직 로그인(Zero Trust) 메뉴를 엽니다. 메뉴 이름은 버전에 따라 다를 수 있습니다.
2. 전달받은 조직 이름으로 등록하고, 브라우저에서 안내된 계정으로 인증합니다.
3. 클라이언트에 올바른 조직이 표시되는지와 연결 상태를 확인합니다.

개인용 WARP 연결이나 다른 기관 조직 연결과 구별해 주세요. 자세한 화면은 [Cloudflare 수동 등록 안내](https://developers.cloudflare.com/cloudflare-one/team-and-resources/devices/cloudflare-one-client/deployment/manual-deployment/)를 참고할 수 있습니다.

## 3. SSH 접속 확인

```bash
# 내 컴퓨터에서 실행
ssh <username>@<server-address>
```

계정·주소·별도 포트는 관리자에게 받은 값을 사용합니다. 접속하면 [처음 시작하기](/ko/server/getting-started)로 돌아가 초기 비밀번호 변경과 첫 GPU 작업을 진행하세요.

## 연결되지 않을 때

| 증상 | 먼저 확인할 내용 |
| --- | --- |
| 연결 시간 초과 | 올바른 조직에 등록했는지, WARP 연결 상태, 서버 주소·포트 |
| `Permission denied` | 사용자 이름과 비밀번호 또는 SSH 키 |
| 갑자기 연결이 끊김 | 다른 VPN과 충돌하는지, 클라이언트 상태, 점검 공지 |

해결되지 않으면 OS, 발생 시각, 에러 메시지와 클라이언트 연결 상태를 관리자에게 보내 주세요. 비밀번호·개인키는 보내지 않습니다.

이 연결 설정은 LAAL 서버용입니다. GSDS 접근 경로는 GSDS의 별도 안내를 따릅니다.
