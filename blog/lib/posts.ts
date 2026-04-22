import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import remarkHtml from "remark-html";
import remarkGfm from "remark-gfm";

const POSTS_DIR = path.join(process.cwd(), "content", "posts");

export type PostMeta = {
  slug: string;
  title: string;
  date: string;
  topic: string;
  excerpt: string;
  paid?: boolean;
  readTime?: string;
};

export type Post = PostMeta & {
  html: string;
  raw: string;
};

export function getAllPostMeta(): PostMeta[] {
  if (!fs.existsSync(POSTS_DIR)) return [];
  const files = fs.readdirSync(POSTS_DIR).filter((f) => f.endsWith(".md"));
  const posts = files.map((file) => {
    const slug = file.replace(/\.md$/, "");
    const raw = fs.readFileSync(path.join(POSTS_DIR, file), "utf8");
    const { data, content } = matter(raw);
    return {
      slug,
      title: data.title ?? slug,
      date: data.date ?? "",
      topic: data.topic ?? "misc",
      excerpt: data.excerpt ?? content.slice(0, 180).replace(/\n/g, " "),
      paid: Boolean(data.paid),
      readTime: estimateReadTime(content),
    } satisfies PostMeta;
  });
  return posts.sort((a, b) => (a.date < b.date ? 1 : -1));
}

export async function getPost(slug: string): Promise<Post | null> {
  const file = path.join(POSTS_DIR, `${slug}.md`);
  if (!fs.existsSync(file)) return null;
  const raw = fs.readFileSync(file, "utf8");
  const { data, content } = matter(raw);
  const processed = await remark().use(remarkGfm).use(remarkHtml).process(content);
  return {
    slug,
    title: data.title ?? slug,
    date: data.date ?? "",
    topic: data.topic ?? "misc",
    excerpt: data.excerpt ?? content.slice(0, 180),
    paid: Boolean(data.paid),
    readTime: estimateReadTime(content),
    html: processed.toString(),
    raw: content,
  };
}

export function getTopics(): { topic: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const post of getAllPostMeta()) {
    counts.set(post.topic, (counts.get(post.topic) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([topic, count]) => ({ topic, count }))
    .sort((a, b) => b.count - a.count);
}

function estimateReadTime(content: string): string {
  const words = content.trim().split(/\s+/).length;
  const mins = Math.max(1, Math.round(words / 220));
  return `${mins} min read`;
}

export function formatDate(date: string): string {
  if (!date) return "";
  const d = new Date(date);
  if (Number.isNaN(d.valueOf())) return date;
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}
