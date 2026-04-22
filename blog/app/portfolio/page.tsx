import Link from "next/link";

export const metadata = { title: "Portfolio — dog with a blog" };

type Project = {
  title: string;
  role: string;
  year: string;
  summary: string;
  link?: string;
  tags: string[];
};

const projects: Project[] = [
  {
    title: "Graduate research — (incoming, fall)",
    role: "Researcher / TA",
    year: "2026",
    summary:
      "Starting grad school in the fall. This section will fill in with syllabi, reading notes, and TA materials.",
    tags: ["academia", "research"],
  },
  {
    title: "dog with a blog",
    role: "Founder, writer, designer",
    year: "2026",
    summary:
      "The site you're on. Built to own my writing end-to-end, separate from the personal feeds, with a paid-subscriber tier.",
    link: "/",
    tags: ["writing", "product"],
  },
  {
    title: "Substack archive (legacy)",
    role: "Writer",
    year: "2024—",
    summary:
      "Essays originally published on Substack, now being ported and rewritten for this site.",
    tags: ["writing"],
  },
];

export default function PortfolioPage() {
  return (
    <div>
      <div className="chip text-accent border-accent/40">Portfolio</div>
      <h1 className="mt-5 font-display text-4xl md:text-5xl font-semibold">
        Projects, research, and side work.
      </h1>
      <p className="mt-3 max-w-2xl text-muted-foreground">
        A living list of the things I&apos;m building, researching, or
        publishing. Some are personal, some academic, some just experiments.
      </p>

      <ul className="mt-10 space-y-6">
        {projects.map((p) => (
          <li key={p.title} className="card-surface p-6">
            <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-2">
              <h2 className="font-display text-2xl font-semibold">
                {p.link ? (
                  <Link href={p.link} className="hover:text-accent">
                    {p.title}
                  </Link>
                ) : (
                  p.title
                )}
              </h2>
              <div className="font-ui text-xs uppercase tracking-widest text-muted-foreground">
                {p.role} &middot; {p.year}
              </div>
            </div>
            <p className="mt-3 text-muted-foreground">{p.summary}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {p.tags.map((t) => (
                <span key={t} className="chip text-muted-foreground">
                  {t}
                </span>
              ))}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
