import Link from "next/link";

export const metadata = { title: "Writings — dog with a blog" };

type Work = {
  title: string;
  form: "poem" | "essay" | "short story" | "letter" | "fragment";
  year: string;
  excerpt: string;
  link?: string;
};

const works: Work[] = [
  {
    title: "A letter to the girl I was",
    form: "letter",
    year: "2026",
    excerpt:
      "I don&apos;t owe anyone the version of me that was easier to explain.",
  },
  {
    title: "Notes from the drive home",
    form: "fragment",
    year: "2025",
    excerpt:
      "Half-thoughts collected at red lights. Some of them eventually became essays.",
  },
  {
    title: "The second account",
    form: "essay",
    year: "2026",
    excerpt:
      "On building a softer, quieter public self — and why I want one.",
  },
];

export default function WritingsPage() {
  return (
    <div>
      <div className="font-ui text-xs uppercase tracking-widest text-[var(--accent)] mb-3">
        Writings &amp; other works
      </div>
      <h1 className="font-display text-4xl md:text-5xl font-semibold">
        The rest of it.
      </h1>
      <p className="mt-3 max-w-2xl text-[var(--muted)]">
        Poems, fragments, letters, experiments. Things that don&apos;t fit in
        the articles feed but are still mine.
      </p>

      <ul className="mt-10 space-y-8">
        {works.map((w) => (
          <li key={w.title} className="border-b border-[var(--rule)] pb-8">
            <div className="font-ui text-xs uppercase tracking-widest text-[var(--muted)]">
              {w.form} &middot; {w.year}
            </div>
            <h2 className="mt-1 font-display text-2xl font-semibold">
              {w.link ? (
                <Link href={w.link} className="hover:text-[var(--accent)]">
                  {w.title}
                </Link>
              ) : (
                w.title
              )}
            </h2>
            <p className="mt-2 italic text-[var(--muted)]">&ldquo;{w.excerpt}&rdquo;</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
