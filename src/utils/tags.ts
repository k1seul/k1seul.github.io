import { getCollection } from 'astro:content';
import { PAGES } from '../config';

// Only read sections that are enabled in pages.ts, so disabled (and possibly
// empty) collections neither show up under tags nor trigger build warnings.
const TAGGED_COLLECTIONS = [
    { collection: 'publications', page: 'publications' },
    { collection: 'talks', page: 'talks' },
    { collection: 'projects', page: 'projects' },
    { collection: 'posts', page: 'blog' },
    { collection: 'teaching', page: 'teaching' },
] as const;

async function getTaggedEntries() {
    const active = TAGGED_COLLECTIONS.filter(({ page }) => PAGES[page]?.isActive !== false);
    const groups = await Promise.all(
        active.map(async ({ collection }) =>
            (await getCollection(collection)).map((e: any) => ({ ...e, collection })),
        ),
    );
    return groups.flat();
}

export async function getAllTags() {
    const allEntries = await getTaggedEntries();
    const tags: Record<string, number> = {};

    allEntries.forEach(entry => {
        const entryTags = (entry.data as any).tags || [];
        entryTags.forEach((tag: string) => {
            const normalizedTag = tag.trim().toLowerCase();
            if (normalizedTag) {
                tags[normalizedTag] = (tags[normalizedTag] || 0) + 1;
            }
        });
    });

    return Object.entries(tags)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

export async function getContentByTag(tag: string) {
    const normalizedSearchTag = tag.toLowerCase();

    const filterFn = (entry: any) => {
        const entryTags = (entry.data as any).tags || [];
        return entryTags.some((t: string) => t.toLowerCase() === normalizedSearchTag);
    };

    return (await getTaggedEntries()).filter(filterFn).sort((a, b) => {
        const dateA = new Date((a.data as any).date || 0);
        const dateB = new Date((b.data as any).date || 0);
        return dateB.getTime() - dateA.getTime();
    });
}
