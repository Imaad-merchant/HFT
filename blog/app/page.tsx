import Link from "next/link";
import PostCard from "@/components/PostCard";
import { getAllPostMeta, getTopics } from "@/lib/posts";

export default function HomePage() {
  const posts = getAllPostMeta();
  const [featured, ...rest] = posts;
  const recent = rest.slice(0, 5);
  const topics = getTopics().slice(0, 6);

  return (
    <div>
      <section className="py-6 md:py-10">
        <div className="font-ui text-xs uppercase tracking-widest text-[var(--accent)] mb-3">
          Welcome
        </div>
        <h1 className="font-display text-4xl md:text-6xl font-semibold leading-[1.05] max-w-3xl">
          Essays, articles, and half-finished thoughts — on my own platform.
        </h1>
        <p className="mt-6 max-w-2xl text-lg text-[var(--muted)]">
          A personal, subscription-supported blog from Isha. Independent from any feed,
          owned end-to-end, and a little harder to stumble into than my Instagram.
        </p>
        <div className="mt-6 flex gap-4 font-ui text-sm">
          <Link
            href="/subscribe"
            className="px-5 py-2 bg-[var(--ink)] text-[var(--bg)] rounded-full"
          >
            Subscribe
          </Link>
          <Link href="/articles" className="px-5 py-2 rounded-full border border-[var(--ink)]">
            Browse articles
          </Link>
        </div>
      </section>

      {featured && (
        <section className="mt-12">
          <div className="font-ui text-xs uppercase tracking-widest text-[var(--muted)] mb-4">
            Latest
          </div>
          <PostCard post={featured} size="lg" />
        </section>
      )}

      <div className="mt-12 grid md:grid-cols-3 gap-12">
        <section className="md:col-span-2">
          <div className="font-ui text-xs uppercase tracking-widest text-[var(--muted)] mb-4">
            Recent posts
          </div>
          {recent.length === 0 ? (
            <p className="text-[var(--muted)]">No posts yet — first one coming soon.</p>
          ) : (
            recent.map((p) => <PostCard key={p.slug} post={p} />)
          )}
          <div className="mt-6 font-ui text-sm">
            <Link href="/articles" className="underline underline-offset-4">
              See all articles &rarr;
            </Link>
          </div>
        </section>

        <aside>
          <div className="font-ui text-xs uppercase tracking-widest text-[var(--muted)] mb-4">
            Topics
          </div>
          <ul className="space-y-2 font-ui text-sm">
            {topics.map((t) => (
              <li key={t.topic} className="flex justify-between border-b border-[var(--rule)] py-1">
                <Link href={`/articles?topic=${encodeURIComponent(t.topic)}`} className="capitalize">
                  {t.topic}
                </Link>
                <span className="text-[var(--muted)]">{t.count}</span>
              </li>
            ))}
          </ul>

          <div className="mt-10 p-5 border border-[var(--rule)] rounded-xl">
            <div className="font-display text-xl font-semibold">Why subscribe?</div>
            <p className="mt-2 text-sm text-[var(--muted)]">
              Members get the full archive, paid-only essays, and the occasional voice note.
            </p>
            <Link
              href="/subscribe"
              className="mt-3 inline-block font-ui text-sm underline underline-offset-4"
            >
              Become a member &rarr;
            </Link>
          </div>
        </aside>
      </div>
    </div>
  );
}
