import type { SiteConfig, ThemeConfig, SettingsConfig, UmamiAnalyticsConfig, AnalyticsConfig } from "../types";

export const SITE: SiteConfig = {
    website: "https://k1seul.github.io/",
    author: "Hanseul Kim",
    desc: "Hanseul Kim: M.S./Ph.D. student at LAAL, Seoul National University, working on robot learning.",
    title: "Hanseul Kim",
    ogImage: "og-image.jpg",
    postPerPage: 10,
    favicon: "/favicon.png",
    lang: "en",
};

export const THEME_CONFIG: ThemeConfig = {
    lightAndDark: true,
    themeLight: "light_default",
    themeDark: "dark_slate",
};

export const SETTINGS: SettingsConfig = {
    showTagsInNavbar: false,
    // Turn on together with PAGES.blog once there are posts.
    showRSSInFooter: false,
    addDevToolsInProduction: false,
};

const umami: UmamiAnalyticsConfig = {
    websiteId: "", // e.g., 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
    src: "https://cloud.umami.is/script.js", // Default Umami cloud script URL
}

export const ANALYTICS: AnalyticsConfig = {
    // Google Analytics 4 Measurement ID (e.g., 'G-XXXXXXXXXX')
    ga4Id: "",
    // Umami Analytics configuration
    umami: umami
};
