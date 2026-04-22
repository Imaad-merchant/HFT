"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setMounted(true);
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {}
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle color theme"
      className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-border hover:bg-secondary"
    >
      <span className="font-ui text-sm" aria-hidden>
        {mounted ? (dark ? "☼" : "☾") : "☾"}
      </span>
    </button>
  );
}
