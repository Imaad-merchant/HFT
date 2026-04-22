import Link from "next/link";
import { PostMeta, formatDate } from "@/lib/posts";

export default function PostCard({
  post,
  size = "md",
}: {
  post: PostMeta;
  size?: "sm" | "md" | "lg";
}) {
  const titleClass =
    size === "lg"
      ? "text-3xl md:text-4xl"
      : size === "sm"
        ? "text-lg"
        : "text-2xl";

  return (
    <article className="group py-6 border-b border-border">
      <div className="font-ui text-xs uppercase tracking-widest text-accent mb-2 flex items-center gap-3">
        <span>{post.topic}</span>
        {post.paid && (
          <span className="rounded-full border border-accent px-2 py-0.5 text-accent">
            Members
          </span>
        )}
      </div>
      <h2
        className={`font-display font-semibold ${titleClass} leading-tight`}
      >
        <Link
          href={`/posts/${post.slug}`}
          className="group-hover:text-accent"
        >
          {post.title}
        </Link>
      </h2>
      {post.excerpt && (
        <p className="mt-2 text-muted-foreground">{post.excerpt}</p>
      )}
      <div className="mt-3 font-ui text-xs text-muted-foreground flex gap-3">
        <span>{formatDate(post.date)}</span>
        {post.readTime && <span>&middot; {post.readTime}</span>}
      </div>
    </article>
  );
}
