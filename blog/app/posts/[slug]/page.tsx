import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllPostMeta, getPost, formatDate } from "@/lib/posts";

export async function generateStaticParams() {
  return getAllPostMeta().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}) {
  const post = await getPost(params.slug);
  if (!post) return { title: "Not found" };
  return {
    title: `${post.title} — dog with a blog`,
    description: post.excerpt,
  };
}

export default async function PostPage({
  params,
}: {
  params: { slug: string };
}) {
  const post = await getPost(params.slug);
  if (!post) return notFound();

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
      <div className="flex items-center gap-3 font-ui text-xs uppercase tracking-widest">
        <Link
          href={`/articles?topic=${encodeURIComponent(post.topic)}`}
          className="chip text-accent border-accent/40 capitalize"
        >
          {post.topic}
        </Link>
        {post.paid && <span className="chip text-accent border-accent/40">Members</span>}
      </div>
      <h1 className="mt-5 font-display text-4xl md:text-5xl font-semibold leading-tight">
        {post.title}
      </h1>
      <div className="mt-4 font-ui text-sm text-muted-foreground">
        {formatDate(post.date)}{" "}
        {post.readTime && <>&middot; {post.readTime}</>}
      </div>

      <div
        className="prose prose-lg mt-10"
        dangerouslySetInnerHTML={{ __html: bodyHtml }}
      />

      {gated && (
        <div className="mt-10 card-surface p-6">
          <div className="font-display text-xl font-semibold">
            The rest is for members.
          </div>
          <p className="mt-2 text-muted-foreground">
            Become a member to read the full essay, the archive, and
            subscriber-only notes.
          </p>
          <Link href="/subscribe" className="btn-primary mt-4">
            Subscribe to keep reading
          </Link>
        </div>
      )}

      <div className="mt-14 rule pt-6 font-ui text-sm flex justify-between">
        <Link href="/articles" className="hover:text-accent underline underline-offset-4">
          &larr; All articles
        </Link>
        <Link href="/subscribe" className="hover:text-accent underline underline-offset-4">
          Subscribe &rarr;
        </Link>
      </div>
    </article>
  );
}
