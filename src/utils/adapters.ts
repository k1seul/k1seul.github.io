import type { ListingItem, DetailItem } from "../types";

function formatDate(dateValue: string | Date | undefined): string | undefined {
    if (!dateValue) return undefined;
    const date = typeof dateValue === 'string' ? new Date(dateValue) : dateValue;
    if (isNaN(date.getTime())) return undefined;
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long' });
}

const STATUS_LABELS: Record<string, string> = {
    planning: "In planning",
    ongoing: "Ongoing",
    completed: "Completed",
};

export function getListingItem(entry: any, collection?: string): ListingItem {
    const d = entry.data;

    return {
        title: d.title,
        description: d.description,
        // Seminar terms show their period and status instead of a single month.
        date: d.period ? [d.period, STATUS_LABELS[d.status]].filter(Boolean).join(" · ") : formatDate(d.date),
        authors: d.author,
        extraInput: d.journal || d.event || d.institution,
        tags: d.tags || [],
        externalUrl: d.external_url,
        image: d.image,
    };
}

export function getDetailItem(entry: any, collection: string): DetailItem {
    const listing = getListingItem(entry, collection);
    
    return {
        ...listing,
        backHref: collection === 'posts' ? '/posts' : `/${collection}`,
    };
}