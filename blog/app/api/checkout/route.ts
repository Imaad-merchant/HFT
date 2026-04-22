import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const plan = req.nextUrl.searchParams.get("plan") ?? "monthly";

  const stripeKey = process.env.STRIPE_SECRET_KEY;
  const priceId =
    plan === "annual"
      ? process.env.STRIPE_PRICE_ANNUAL
      : process.env.STRIPE_PRICE_MONTHLY;

  if (!stripeKey || !priceId) {
    return NextResponse.json(
      {
        error: "Stripe not configured",
        hint: "Set STRIPE_SECRET_KEY and STRIPE_PRICE_MONTHLY / STRIPE_PRICE_ANNUAL in .env.local, then wire this handler to stripe.checkout.sessions.create().",
        plan,
      },
      { status: 501 },
    );
  }

  // Wire up a real Stripe Checkout session here, e.g.:
  //
  //   const stripe = new Stripe(stripeKey);
  //   const session = await stripe.checkout.sessions.create({
  //     mode: "subscription",
  //     line_items: [{ price: priceId, quantity: 1 }],
  //     success_url: `${req.nextUrl.origin}/subscribe/success`,
  //     cancel_url: `${req.nextUrl.origin}/subscribe`,
  //   });
  //   return NextResponse.redirect(session.url!, 303);

  return NextResponse.json({ ok: true, plan, priceId });
}
