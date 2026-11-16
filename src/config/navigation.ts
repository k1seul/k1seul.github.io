import type { NavLink } from "../types";

// Links whose section is disabled in pages.ts are hidden automatically.
export const NAV_LINKS: NavLink[] = [
    { href: "/", label: "About", isActive: true },
    { href: "/publications", label: "Publications", isActive: true },
    { href: "/projects", label: "Studies", isActive: true },
    { href: "/seminar", label: "Seminar", isActive: true },
    { href: "/posts", label: "Blog", isActive: true },
    { href: "/talks", label: "Talks", isActive: true },
    { href: "/teaching", label: "Teaching", isActive: true },
    { href: "/tags", label: "Tags", isActive: false },
    { href: "/cv", label: "CV", isActive: true },
    { href: "/server", label: "Server", isActive: true },
];
