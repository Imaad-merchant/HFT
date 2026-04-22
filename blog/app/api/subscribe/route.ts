import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const email = String(form.get("email") ?? "").trim();

  if (!email || !/.+@.+\..+/.test(email)) {
    return NextResponse.redirect(new URL("/subscribe?error=invalid", req.url), 303);
  }

  // Wire to your email provider here (Buttondown, Beehiiv, ConvertKit, Resend audiences, etc.)
  // For now, just log and redirect to a thanks state.
  console.log("[newsletter] signup:", email);

  return NextResponse.redirect(new URL("/subscribe?joined=1", req.url), 303);
}
