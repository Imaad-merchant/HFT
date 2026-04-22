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
      "I don't owe anyone the version of me that was easier to explain.",
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
    excerpt: "On building a softer, quieter public self — and why I want one.",
  },
];

export default function WritingsPage() {
  return (
    <div>
      <div className="chip text-accent border-accent/40">
        Writings &amp; other works
      </div>
      <h1 className="mt-5 font-display text-4xl md:text-5xl font-semibold">
        The rest of it.
      </h1>
      <p className="mt-3 max-w-2xl text-muted-foreground">
        Poems, fragments, letters, experiments. Things that don&apos;t fit in
        the articles feed but are still mine.
      </p>

      <ul className="mt-10 space-y-6">
        {works.map((w) => (
          <li key={w.title} className="card-surface p-6">
            <div className="font-ui text-xs uppercase tracking-widest text-muted-foreground">
              {w.form} &middot; {w.year}
            </div>
            <h2 className="mt-1 font-display text-2xl font-semibold">
              {w.link ? (
                <Link href={w.link} className="hover:text-accent">
                  {w.title}
                </Link>
              ) : (
                w.title
              )}
            </h2>
            <p className="mt-2 italic text-muted-foreground">
              &ldquo;{w.excerpt}&rdquo;
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
