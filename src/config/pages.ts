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
        subtitle: "Self-study notes and from-scratch implementations: VLMs, reinforcement learning, math, and Python.",
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
    server: {
        title: "Lab Server",
        subtitle: "계정 신청부터 첫 GPU 작업까지, 그리고 GSDS 스토리지 운영 계획.",
        isActive: true,
    },
    cv: {
        title: "CV",
        subtitle: "",
        isActive: true,
    },
};

