import Link from "next/link";
import ThemeToggle from "./ThemeToggle";

const links = [
  { href: "/", label: "Home" },
  { href: "/articles", label: "Articles" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/writings", label: "Writings" },
  { href: "/subscribe", label: "Subscribe" },
];

export default function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 glass border-b border-border">
      <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between gap-6">
        <Link
          href="/"
          className="font-display text-xl md:text-2xl font-semibold tracking-tight"
        >
          <span className="text-accent">&bull;</span> dog with a blog
        </Link>
        <div className="flex items-center gap-4 md:gap-6">
          <nav className="hidden md:block font-ui text-xs uppercase tracking-widest">
            <ul className="flex gap-5">
              {links.map((l) => (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    className="text-muted-foreground hover:text-accent"
                  >
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
