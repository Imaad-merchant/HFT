# dog with a blog

A personal, subscription-supported blog site for Isha. Built with Next.js (App
Router) + Tailwind + markdown posts. Lives in this repo under `blog/` so it
doesn't collide with the HFT code at the root.

## Pages

- `/` — recent posts + featured essay + topics sidebar
- `/articles` — all posts grouped by topic, with a topic filter
- `/portfolio` — projects, research, side work
- `/writings` — poems, fragments, letters, non-essay work
- `/subscribe` — free + paid tiers (Stripe wiring is stubbed)
- `/posts/[slug]` — individual post; paid posts show a paywall after two paragraphs

## Writing a post

Drop a markdown file in `content/posts/`:

```md
---
title: "My post title"
date: "2026-04-22"
topic: "personal"       # becomes a tag/section
excerpt: "One-line summary shown in lists."
paid: false             # true = gated after two paragraphs
---

Your post body, in markdown.
```

## Running locally

```bash
cd blog
npm install
npm run dev
```

Then open http://localhost:3000.

## Wiring payments

The subscribe page links to stub API routes:

- `app/api/checkout/route.ts` — create a Stripe Checkout session here
- `app/api/subscribe/route.ts` — forward email to your newsletter provider

Copy `.env.example` to `.env.local` and fill in the keys. The paywall on
`/posts/[slug]` is purely display-layer today — once you have auth + Stripe
set up, check the user's subscription status server-side and return the full
HTML when they're a paid member.

## Deploying to Vercel

The site is configured via `vercel.json` (framework pinned to Next.js, sensible
security headers). Two ways to deploy:

**Dashboard:** vercel.com/new → import this repo → Vercel auto-detects Next.js
and deploys. If this is a monorepo (e.g. still inside the HFT repo), set
**Root Directory** to `blog`.

**CLI:**

```bash
npx vercel login
npx vercel            # preview
npx vercel --prod     # production
```

### Preview deployments on PRs

`.github/workflows/preview.yml` deploys a Vercel preview for every PR and
posts the URL as a sticky comment. Add these repo secrets in GitHub
(Settings → Secrets and variables → Actions):

- `VERCEL_TOKEN` — create at vercel.com/account/tokens
- `VERCEL_ORG_ID` — from `.vercel/project.json` after the first `vercel` run, or Project → Settings → General
- `VERCEL_PROJECT_ID` — same source as above

Vercel's built-in Git integration also comments preview URLs automatically if
you connect the repo through the Vercel dashboard — this workflow is useful
if you'd rather run the deploy from CI than from Vercel's side.
