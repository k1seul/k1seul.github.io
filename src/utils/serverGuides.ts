import { getCollection, type CollectionEntry } from "astro:content";
import { serverHref, type ServerLanguage, type ServerSectionId } from "../config/server";

export function serverGuideSlug(guide: CollectionEntry<"server">) {
    return guide.id.split("/").at(-1)!;
}

export function serverGuideHref(guide: CollectionEntry<"server">) {
    return serverHref("/server/" + serverGuideSlug(guide), guide.data.language);
}

export async function getServerGuides(language: ServerLanguage, section?: ServerSectionId) {
    return (await getCollection("server"))
        .filter((guide) => guide.data.language === language && (!section || guide.data.section === section))
        .sort((a, b) => a.data.order - b.data.order);
}
