import Link from "next/link";

export default function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-border">
      <div className="mx-auto max-w-5xl px-6 py-10 flex flex-col md:flex-row justify-between gap-6 font-ui text-sm text-muted-foreground">
        <div>
          <div className="font-display text-lg text-foreground">
            <span className="text-accent">&bull;</span> dog with a blog
          </div>
          <div>Owned, written, and published by Isha.</div>
        </div>
        <div className="flex flex-wrap gap-5">
          <Link href="/articles" className="hover:text-accent">
            Articles
          </Link>
          <Link href="/portfolio" className="hover:text-accent">
            Portfolio
          </Link>
          <Link href="/writings" className="hover:text-accent">
            Writings
          </Link>
          <Link href="/subscribe" className="hover:text-accent">
            Subscribe
          </Link>
        </div>
        <div>&copy; {new Date().getFullYear()}</div>
      </div>
    </footer>
  );
}
