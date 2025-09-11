"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export default function Home() {
  const [clientName, setClientName] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projectType, setProjectType] = useState("General");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<{ type: "idle"|"loading"|"success"|"error"; message?: string }>({ type: "idle" });
  const [uploadedPath, setUploadedPath] = useState<string>("");

  async function handleUpload(): Promise<string> {
    if (!file) throw new Error("Please select a PDF file");
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_BASE}/upload_pdf`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || "Upload failed");
    }
    const data = await res.json();
    return data.path as string;
  }

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus({ type: "loading", message: "Uploading PDF…" });
    try {
      const path = await handleUpload();
      setUploadedPath(path);
      setStatus({ type: "loading", message: "Starting proposal generation…" });

      const res = await fetch(`${API_BASE}/generate_proposal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pdf_path: path,
          client_name: clientName,
          project_name: projectName,
          project_type: projectType,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Failed to queue generation");
      setStatus({ type: "success", message: data.message || "Queued" });
    } catch (err: any) {
      setStatus({ type: "error", message: err?.message || "Something went wrong" });
    }
  }

  return (
    <div className="container">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark" />
          <div>
            <div className="brand-name">Proposal Generator</div>
            <div className="muted" style={{ fontSize: 12 }}>Create comprehensive, professional proposals</div>
          </div>
        </div>
        <div className="chip info">v2.0</div>
      </header>

      <main style={{ paddingTop: 24, paddingBottom: 24 }}>
        <section className="card" style={{ padding: 24, marginBottom: 16 }}>
          <div className="h1" style={{ marginBottom: 8 }}>New Proposal</div>
          <div className="muted" style={{ marginBottom: 20 }}>Fill in details and upload a PDF RFP (optional)</div>

          <form onSubmit={onSubmit} className="form">
            <div className="form-grid" style={{ marginBottom: 16 }}>
              <div className="form-row">
                <label>Client name</label>
                <input type="text" value={clientName} onChange={(e) => setClientName(e.target.value)} placeholder="Acme Corp" required />
              </div>
              <div className="form-row">
                <label>Project name</label>
                <input type="text" value={projectName} onChange={(e) => setProjectName(e.target.value)} placeholder="AI Transformation" required />
              </div>
              <div className="form-row">
                <label>Project type</label>
                <select value={projectType} onChange={(e) => setProjectType(e.target.value)}>
                  <option>General</option>
                  <option>Healthcare</option>
                  <option>Government</option>
                  <option>Finance</option>
                  <option>Education</option>
                  <option>Technology</option>
                </select>
              </div>
              <div className="form-row">
                <label>RFP PDF</label>
                <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              </div>
            </div>

            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <button className="btn btn-primary" type="submit" disabled={status.type === "loading"}>
                {status.type === "loading" ? "Processing…" : "Generate Proposal"}
              </button>
              {uploadedPath && <span className="chip info">Uploaded: {uploadedPath.split("/").pop()}</span>}
              {status.type === "success" && <span className="chip success">{status.message}</span>}
              {status.type === "error" && <span className="chip error">{status.message}</span>}
            </div>
          </form>
        </section>

        <section className="card" style={{ padding: 24 }}>
          <div className="h2" style={{ marginBottom: 8 }}>Tips</div>
          <ul className="muted" style={{ marginLeft: 18, lineHeight: 1.8 }}>
            <li>Use clear client and project names for organized outputs.</li>
            <li>Upload an RFP PDF to guide the generator with RAG.</li>
            <li>While queued, check backend logs for progress.</li>
          </ul>
        </section>
      </main>

      <footer className="app-footer">© {new Date().getFullYear()} Proposal Generator</footer>
    </div>
  );
}
