export type ServerLanguage = "en" | "ko";
export type ServerSectionId = "laal" | "gsds";

export function serverHref(path: string, language: ServerLanguage = "en") {
    return language === "ko" ? "/ko" + path : path;
}

export const SERVER_COPY = {
    en: {
        title: "Lab Server",
        subtitle: "Get started with LAAL servers and find information about GSDS storage access.",
        quickNav: "Quick links", previous: "Previous guide", next: "Next guide",
        quick: {
            laal: ["Getting started", "Run a GPU job", "Troubleshooting"],
            gsds: ["Service overview", "Apply for access", "Check your account"],
        },
        sections: {
            laal: { label: "LAAL Servers", intro: "New here? Start with account setup and your first GPU job. Hanseul Kim and Minsoo Jo maintain the servers. Request server addresses and access details directly from the admins." },
            gsds: { label: "GSDS Storage", intro: "Not open yet. Learn who can use the planned LAAL storage at GSDS, how to apply, how accounts will be registered, and how to check your access after launch." },
        },
    },
    ko: {
        title: "연구실 서버 안내",
        subtitle: "LAAL 서버 시작 방법과 GSDS 스토리지 이용·계정 신청 안내입니다.",
        quickNav: "빠른 안내", previous: "이전 안내", next: "다음 안내",
        quick: {
            laal: ["처음 시작하기", "GPU 작업 실행", "문제 해결"],
            gsds: ["서비스 안내", "이용 신청", "내 계정 확인"],
        },
        sections: {
            laal: { label: "LAAL 서버 이용", intro: "처음이라면 계정 신청과 첫 GPU 작업부터 따라가세요. 김한슬·조민수가 관리하며, 서버 주소와 접속 정보는 관리자에게 별도로 안내받습니다." },
            gsds: { label: "GSDS 스토리지", intro: "아직 개통 전입니다. 이용 대상, 신청 방법, 계정 등록 절차와 개통 후 권한을 확인하는 방법을 안내합니다." },
        },
    },
} as const;

export function getServerSections(language: ServerLanguage = "en") {
    return (["laal", "gsds"] as const).map((id) => ({
        id,
        href: serverHref(id === "laal" ? "/server" : "/server/gsds", language),
        ...SERVER_COPY[language].sections[id],
    }));
}

