import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const { apiKey } = await request.json();
    if (!apiKey) {
      return NextResponse.json({ ok: false, error: "Missing apiKey" }, { status: 400 });
    }

    const res = await fetch("https://api.openai.com/v1/models", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    });

    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json({ ok: false, error: text || "Invalid key or network" }, { status: res.status });
    }

    return NextResponse.json({ ok: true });
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: error?.message || "Failed to verify" }, { status: 500 });
  }
}


