import Link from "next/link";

export default function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-[var(--rule)]">
      <div className="mx-auto max-w-5xl px-6 py-10 flex flex-col md:flex-row justify-between gap-6 font-ui text-sm text-[var(--muted)]">
        <div>
          <div className="font-display text-lg text-[var(--ink)]">dog with a blog</div>
          <div>Owned, written, and published by Isha.</div>
        </div>
        <div className="flex gap-6">
          <Link href="/articles">Articles</Link>
          <Link href="/portfolio">Portfolio</Link>
          <Link href="/writings">Writings</Link>
          <Link href="/subscribe">Subscribe</Link>
        </div>
        <div>&copy; {new Date().getFullYear()}</div>
      </div>
    </footer>
  );
}
