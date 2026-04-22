import Link from "next/link";

export const metadata = { title: "Subscribe — dog with a blog" };

const tiers = [
  {
    name: "Free",
    price: "$0",
    cadence: "",
    perks: [
      "Public essays as they publish",
      "Monthly newsletter",
    ],
    cta: "Stay free",
    ctaHref: "#signup",
    highlight: false,
  },
  {
    name: "Member",
    price: "$6",
    cadence: "/ month",
    perks: [
      "Everything in Free",
      "Full archive + subscriber-only essays",
      "Occasional voice notes",
      "Comment on posts",
    ],
    cta: "Become a member",
    ctaHref: "/api/checkout?plan=monthly",
    highlight: true,
  },
  {
    name: "Annual",
    price: "$60",
    cadence: "/ year",
    perks: [
      "Everything in Member",
      "Two months free",
      "First read on long-form pieces",
    ],
    cta: "Subscribe annually",
    ctaHref: "/api/checkout?plan=annual",
    highlight: false,
  },
];

export default function SubscribePage() {
  return (
    <div>
      <div className="font-ui text-xs uppercase tracking-widest text-[var(--accent)] mb-3">
        Subscribe
      </div>
      <h1 className="font-display text-4xl md:text-5xl font-semibold">
        Keep this thing going.
      </h1>
      <p className="mt-3 max-w-2xl text-[var(--muted)]">
        This site is independent. Subscribing is how I keep writing here
        instead of handing it to someone else&apos;s platform.
      </p>

      <div className="mt-10 grid md:grid-cols-3 gap-6">
        {tiers.map((t) => (
          <div
            key={t.name}
            className={`p-6 border rounded-2xl ${
              t.highlight
                ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--bg)]"
                : "border-[var(--rule)]"
            }`}
          >
            <div className="font-ui text-xs uppercase tracking-widest opacity-70">
              {t.name}
            </div>
            <div className="mt-2 font-display text-4xl font-semibold">
              {t.price}
              <span className="text-base font-normal opacity-70"> {t.cadence}</span>
            </div>
            <ul className="mt-4 space-y-2 text-sm">
              {t.perks.map((p) => (
                <li key={p}>&middot; {p}</li>
              ))}
            </ul>
            <Link
              href={t.ctaHref}
              className={`mt-6 inline-block px-5 py-2 rounded-full font-ui text-sm ${
                t.highlight
                  ? "bg-[var(--bg)] text-[var(--ink)]"
                  : "bg-[var(--ink)] text-[var(--bg)]"
              }`}
            >
              {t.cta}
            </Link>
          </div>
        ))}
      </div>

      <section id="signup" className="mt-16 p-6 md:p-10 border border-[var(--rule)] rounded-2xl">
        <div className="font-ui text-xs uppercase tracking-widest text-[var(--muted)]">
          Free newsletter
        </div>
        <h2 className="mt-2 font-display text-2xl md:text-3xl font-semibold">
          Just want the emails?
        </h2>
        <p className="mt-2 text-[var(--muted)]">
          Drop your email and I&apos;ll send new posts when they go up. No spam,
          no tracking pixels.
        </p>
        <form
          action="/api/subscribe"
          method="post"
          className="mt-5 flex flex-col sm:flex-row gap-3 max-w-lg"
        >
          <input
            type="email"
            name="email"
            required
            placeholder="you@example.com"
            className="flex-1 px-4 py-2 border border-[var(--rule)] rounded-full bg-transparent font-ui"
          />
          <button
            type="submit"
            className="px-5 py-2 bg-[var(--ink)] text-[var(--bg)] rounded-full font-ui text-sm"
          >
            Sign up
          </button>
        </form>
      </section>

      <p className="mt-10 text-xs text-[var(--muted)] font-ui">
        Paid checkout routes are stubs — wire up Stripe (or Lemon Squeezy / Memberful)
        in <code>app/api/checkout/route.ts</code> and <code>app/api/subscribe/route.ts</code>.
      </p>
    </div>
  );
}
