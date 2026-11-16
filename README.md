# k1seul.github.io

Personal academic site of Hanseul Kim, built with Astro on top of [Academic Portfolio Astro](https://github.com/rubzip/academic-portfolio-astro) (MIT).

## Run locally

Requires Node.js >= 22.12.

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # static output in ./dist
```

## Where things live

| What | File |
| --- | --- |
| About page + sidebar (photo, lab, short bio) | `src/content/bio.md` |
| CV | `src/content/cv.md` |
| Publications | `src/content/publications/*.md` |
| Studies | `src/content/projects/*.md` |
| Blog posts | `src/content/posts/*.md` |
| Social links | `src/config/social.ts` |
| Turn sections on/off (Blog, Talks, Teaching, …) | `src/config/pages.ts` |
| Colors / light & dark themes | `src/config/themes.ts`, `src/config/site.ts` |
| Images and files (PDFs, photos) | `public/` (e.g. `public/profile.jpg` → `/profile.jpg`) |

To start the blog: add a `.md` file under `src/content/posts/`, then set `blog.isActive: true` in `pages.ts` (and `showRSSInFooter: true` in `site.ts` if you want the RSS link).

Math is written in LaTeX (`$...$`, `$$...$$`) and rendered with KaTeX.

## Deploy

Pushing to `main` (or `master`) runs `.github/workflows/deploy.yml`, which builds and publishes to GitHub Pages.
In the repo settings, **Pages → Build and deployment → Source** must be set to **GitHub Actions**.
