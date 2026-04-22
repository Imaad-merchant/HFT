import type { Metadata } from "next";
import "./globals.css";
import SiteHeader from "@/components/SiteHeader";
import SiteFooter from "@/components/SiteFooter";
import ThemeScript from "@/components/ThemeScript";

export const metadata: Metadata = {
  title: "dog with a blog — Isha",
  description:
    "A personal, subscription-supported blog. Essays, articles, portfolio, and writings — owned and published by Isha.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <ThemeScript />
      </head>
      <body className="min-h-screen bg-background text-foreground">
        <SiteHeader />
        <main className="mx-auto max-w-5xl px-6 py-14 fade-in">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}
