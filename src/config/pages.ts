import type { PagesConfig } from "../types";

// Flip `isActive` to show/hide a whole section (page + navbar link).
export const PAGES: PagesConfig = {
    home: {
        title: "About",
        subtitle: "",
        isActive: true,
    },
    blog: {
        title: "Blog",
        subtitle: "Notes on reinforcement learning, deep learning, and math.",
        isActive: false,
    },
    publications: {
        title: "Publications",
        subtitle: "",
        isActive: true,
    },
    talks: {
        title: "Talks",
        subtitle: "",
        isActive: false,
    },
    projects: {
        title: "Studies",
        subtitle: "Self-study notes and code on math, reinforcement learning, and Python.",
        isActive: true,
    },
    teaching: {
        title: "Teaching",
        subtitle: "",
        isActive: false,
    },
    tags: {
        title: "Tags",
        subtitle: "Explore content by topic.",
        isActive: true,
    },
    cv: {
        title: "CV",
        subtitle: "",
        isActive: true,
    },
};
