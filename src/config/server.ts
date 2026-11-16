// Sections of the Server tab. Each section is one sub-tab with its own guide list.
export const SERVER_SECTIONS = [
    {
        id: "laal",
        label: "LAAL Servers",
        href: "/server",
        intro: "Guides for LAAL members, maintained by the server admins (Hanseul Kim and Minsoo Jo). Server addresses, ports, and passwords are not posted here. Ask the admins for access.",
    },
    {
        id: "gsds",
        label: "GSDS Storage",
        href: "/server/gsds",
        intro: "LAAL plans to run its own storage server in the GSDS server room. This section covers the plan, how to request space, quotas, and who manages it. It is a plan, so details may change before launch.",
    },
] as const;

export type ServerSectionId = (typeof SERVER_SECTIONS)[number]["id"];
