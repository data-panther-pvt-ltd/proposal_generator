import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    if (!file || !(file instanceof File)) {
      return NextResponse.json({ error: "Missing file" }, { status: 400 });
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const uploadsDir = path.resolve(process.cwd(), "..", "uploads");
    await fs.mkdir(uploadsDir, { recursive: true });

    // Ensure a safe filename
    const originalName = file.name || "upload.pdf";
    const safeName = originalName.replace(/[^a-zA-Z0-9_.-]/g, "_");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${timestamp}_${safeName}`;
    const destPath = path.join(uploadsDir, filename);

    await fs.writeFile(destPath, buffer);

    return NextResponse.json({
      ok: true,
      savedPath: destPath,
      filename,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error?.message || "Failed to save file" },
      { status: 500 }
    );
  }
}


