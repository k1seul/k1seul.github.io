import type { SocialLink } from "../types";

export const SOCIALS: SocialLink[] = [
    {
        name: "Github",
        href: "https://github.com/k1seul",
        linkTitle: "Hanseul Kim on GitHub",
        isActive: true,
    },
    {
        name: "Google Scholar",
        href: "https://scholar.google.com/citations?user=nSkU6zgAAAAJ&hl=ko",
        linkTitle: "Hanseul Kim on Google Scholar",
        isActive: true,
    },
    {
        name: "LinkedIn",
        href: "https://www.linkedin.com/in/k1seul/",
        linkTitle: "Hanseul Kim on LinkedIn",
        isActive: true,
    },
];

export const SOCIAL_ICONS: Record<string, string> = {
    Github: "Github",
    Mail: "Mail",
    LinkedIn: "LinkedIn",
    "Google Scholar": "GoogleScholar",
    ORCID: "ORCID",
    RSS: "RSS",
};
