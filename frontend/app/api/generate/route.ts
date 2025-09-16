import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { pdf_path, client_name, project_name, project_type, backendUrl } = body || {};

    if (!pdf_path) {
      return NextResponse.json({ error: "pdf_path is required" }, { status: 400 });
    }

    const apiBase = backendUrl || process.env.BACKEND_URL || "http://34.28.203.178:8003";

    const res = await fetch(`${apiBase}/generate_proposal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pdf_path,
        client_name: client_name || "",
        project_name: project_name || "",
        project_type: project_type || "general",
      }),
    });

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || "Failed to call backend" },
      { status: 500 }
    );
  }
}


