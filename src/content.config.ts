import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'zod';

const publications = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/publications" }),
    schema: z.object({
        title: z.string(),
        author: z.string().optional(),
        date: z.string().optional(),
        journal: z.string().optional(),
        external_url: z.string().optional(),
        image: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
    }),
});

const talks = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/talks" }),
    schema: z.object({
        title: z.string(),
        date: z.string().optional(),
        event: z.string().optional(),
        external_url: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
        image: z.string().optional(),
    }),
});

const posts = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
    schema: z.object({
        title: z.string(),
        date: z.string().optional(),
        description: z.string().optional(),
        author: z.string().optional(),
        tags: z.array(z.string()).optional(),
        external_url: z.string().optional(),
        image: z.string().optional(),
    }),
});

const teaching = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/teaching" }),
    schema: z.object({
        title: z.string(),
        institution: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
        external_url: z.string().url().optional(),
    }),
});

const bio = defineCollection({
    loader: glob({ pattern: "bio.md", base: "./src/content" }),
    schema: z.object({
        name: z.string(),
        avatar: z.string(),
        avatarHover: z.string().optional(),
        shortBio: z.string().optional(),
        institution: z.string().optional(),
        institutionUrl: z.string().url().optional(),
    }),
});

const projects = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/projects" }),
    schema: z.object({
        title: z.string(),
        date: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
        external_url: z.string().optional(),
        image: z.string().optional(),
    }),
});

// Lab server guides (Server tab), grouped by `section` and listed by `order`
// section "laal": LAAL cluster guides (/server)
// section "gsds": LAAL storage at GSDS (/server/gsds)
const server = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/server" }),
    schema: z.object({
        title: z.string(),
        description: z.string().optional(),
        section: z.enum(["laal", "gsds"]).default("laal"),
        language: z.enum(["en", "ko"]).default("en"),
        order: z.number().default(100),
        date: z.string().optional(), // last updated
    }),
});

// LAAL Robotics Seminar (Seminar tab): one file per term.
// `sessions` drives the schedule rendered under the markdown body.
const seminarPaper = z.object({
    title: z.string(),
    url: z.string().url().optional(),
    venue: z.string().optional(),
    idea: z.string().optional(),   // one-line main idea
    slides: z.string().url().optional(),
    video: z.string().url().optional(),
});

const seminar = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/seminar" }),
    schema: z.object({
        title: z.string(),
        date: z.string(),            // term start, used for sorting
        period: z.string(),          // human-readable, e.g. "July – September 2026"
        status: z.enum(["planning", "ongoing", "completed"]),
        institution: z.string().optional(),
        description: z.string().optional(),
        image: z.string().optional(),
        sessions: z.array(z.object({
            week: z.union([z.number(), z.string()]),
            date: z.string().optional(),
            track: z.string().optional(),
            topic: z.string().optional(),
            image: z.string().optional(),
            papers: z.array(seminarPaper).default([]),
        })).default([]),
    }),
});

const cv = defineCollection({
    loader: glob({ pattern: "cv.md", base: "./src/content" }),
    schema: z.object({
        name: z.string(),
        title: z.string(),
        experience: z.array(z.object({
            role: z.string(),
            institution: z.string(),
            period: z.string(),
            description: z.string(),
            url: z.string().url().optional(),
            advisor: z.string().optional(),
            advisorUrl: z.string().url().optional(),
        })).optional(),
        education: z.array(z.object({
            degree: z.string(),
            institution: z.string(),
            period: z.string(),
            thesis: z.string().optional(),
            description: z.string().optional(),
            url: z.string().url().optional(),
            advisor: z.string().optional(),
            advisorUrl: z.string().url().optional(),
        })).optional(),
    }),
});

export const collections = {
    'publications': publications,
    'talks': talks,
    'posts': posts,
    'bio': bio,
    'projects': projects,
    'cv': cv,
    'teaching': teaching,
    'server': server,
    'seminar': seminar,
};

