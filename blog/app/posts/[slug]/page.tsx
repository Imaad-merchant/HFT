import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllPostMeta, getPost, formatDate } from "@/lib/posts";

export async function generateStaticParams() {
  return getAllPostMeta().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug);
  if (!post) return { title: "Not found" };
  return {
    title: `${post.title} — dog with a blog`,
    description: post.excerpt,
  };
}

export default async function PostPage({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug);
  if (!post) return notFound();

  // Paywall: show first two paragraphs of HTML, then gate the rest for paid posts.
  let bodyHtml = post.html;
  let gated = false;
  if (post.paid) {
    const paragraphs = post.html.split("</p>");
    if (paragraphs.length > 2) {
      bodyHtml = paragraphs.slice(0, 2).join("</p>") + "</p>";
      gated = true;
    }
  }

  return (
    <article className="mx-auto max-w-2xl">
      <div className="font-ui text-xs uppercase tracking-widest text-[var(--accent)] mb-3 flex gap-3">
        <Link href={`/articles?topic=${encodeURIComponent(post.topic)}`} className="capitalize">
          {post.topic}
        </Link>
        {post.paid && (
          <span className="px-2 py-0.5 border border-[var(--accent)] rounded-full">
            Subscribers
          </span>
        )}
      </div>
      <h1 className="font-display text-4xl md:text-5xl font-semibold leading-tight">
        {post.title}
      </h1>
      <div className="mt-4 font-ui text-sm text-[var(--muted)]">
        {formatDate(post.date)} {post.readTime && <>&middot; {post.readTime}</>}
      </div>

      <div
        className="prose-body mt-10"
        dangerouslySetInnerHTML={{ __html: bodyHtml }}
      />

      {gated && (
        <div className="mt-10 p-6 border border-[var(--rule)] rounded-xl bg-white/40">
          <div className="font-display text-xl font-semibold">
            The rest is for subscribers.
          </div>
          <p className="mt-2 text-[var(--muted)]">
            Become a member to read the full essay, the archive, and subscriber-only
            notes.
          </p>
          <Link
            href="/subscribe"
            className="mt-4 inline-block px-5 py-2 bg-[var(--ink)] text-[var(--bg)] rounded-full font-ui text-sm"
          >
            Subscribe to keep reading
          </Link>
        </div>
      )}

      <div className="mt-14 rule pt-6 font-ui text-sm flex justify-between">
        <Link href="/articles" className="underline underline-offset-4">
          &larr; All articles
        </Link>
        <Link href="/subscribe" className="underline underline-offset-4">
          Subscribe &rarr;
        </Link>
      </div>
    </article>
  );
}
