import Link from "next/link";

const links = [
  { href: "/", label: "Home" },
  { href: "/articles", label: "Articles" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/writings", label: "Writings" },
  { href: "/subscribe", label: "Subscribe" },
];

export default function SiteHeader() {
  return (
    <header className="border-b border-[var(--rule)]">
      <div className="mx-auto max-w-5xl px-6 py-6 flex items-center justify-between">
        <Link href="/" className="font-display text-2xl font-semibold tracking-tight">
          dog with a blog
        </Link>
        <nav className="font-ui text-sm uppercase tracking-widest">
          <ul className="flex gap-6">
            {links.map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="hover:text-[var(--accent)]">
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
