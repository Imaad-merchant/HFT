import Link from "next/link";
import PostCard from "@/components/PostCard";
import { getAllPostMeta, getTopics } from "@/lib/posts";

export const metadata = { title: "Articles — dog with a blog" };

export default function ArticlesPage({
  searchParams,
}: {
  searchParams?: { topic?: string };
}) {
  const activeTopic = searchParams?.topic?.toLowerCase();
  const posts = getAllPostMeta();
  const filtered = activeTopic ? posts.filter((p) => p.topic.toLowerCase() === activeTopic) : posts;
  const topics = getTopics();

  const grouped = new Map<string, typeof posts>();
  for (const p of filtered) {
    const arr = grouped.get(p.topic) ?? [];
    arr.push(p);
    grouped.set(p.topic, arr);
  }

  return (
    <div>
      <div className="font-ui text-xs uppercase tracking-widest text-[var(--accent)] mb-3">
        Articles
      </div>
      <h1 className="font-display text-4xl md:text-5xl font-semibold">
        {activeTopic ? `#${activeTopic}` : "Everything, sorted by topic"}
      </h1>
      <p className="mt-3 max-w-2xl text-[var(--muted)]">
        Browse by subject. Paid-only essays are marked and open for members.
      </p>

      <div className="mt-8 flex flex-wrap gap-2 font-ui text-xs uppercase tracking-widest">
        <Link
          href="/articles"
          className={`px-3 py-1 rounded-full border ${
            !activeTopic
              ? "bg-[var(--ink)] text-[var(--bg)] border-[var(--ink)]"
              : "border-[var(--rule)]"
          }`}
        >
          All
        </Link>
        {topics.map((t) => (
          <Link
            key={t.topic}
            href={`/articles?topic=${encodeURIComponent(t.topic)}`}
            className={`px-3 py-1 rounded-full border capitalize ${
              activeTopic === t.topic.toLowerCase()
                ? "bg-[var(--ink)] text-[var(--bg)] border-[var(--ink)]"
                : "border-[var(--rule)]"
            }`}
          >
            {t.topic} <span className="opacity-60">({t.count})</span>
          </Link>
        ))}
      </div>

      <div className="mt-10 space-y-12">
        {[...grouped.entries()].map(([topic, items]) => (
          <section key={topic}>
            <h2 className="font-display text-2xl font-semibold capitalize border-b border-[var(--rule)] pb-2">
              {topic}
            </h2>
            <div>
              {items.map((p) => (
                <PostCard key={p.slug} post={p} />
              ))}
            </div>
          </section>
        ))}
        {filtered.length === 0 && (
          <p className="text-[var(--muted)]">Nothing here yet for this topic.</p>
        )}
      </div>
    </div>
  );
}
