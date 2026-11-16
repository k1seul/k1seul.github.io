// Sections of the Server tab. Each section is one sub-tab with its own guide list.
export const SERVER_SECTIONS = [
    {
        id: "laal",
        label: "LAAL 서버 이용",
        href: "/server",
        intro: "처음이라면 ‘처음 시작하기’에서 계정 신청과 첫 GPU 작업을 따라가세요. 김한슬·조민수가 관리하며, 서버 주소·포트·비밀번호는 관리자에게 별도로 안내받습니다.",
    },
    {
        id: "gsds",
        label: "GSDS 스토리지 계획",
        href: "/server/gsds",
        intro: "아직 개통 전입니다. LAAL이 소유·운영할 GSDS 내 스토리지의 구성, 신청 정책과 유지보수 계획을 안내합니다. 랙·네트워크·전원은 GSDS와 협의 후 확정합니다.",
    },
] as const;

export type ServerSectionId = (typeof SERVER_SECTIONS)[number]["id"];

