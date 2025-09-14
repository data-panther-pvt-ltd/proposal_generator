"use client";

import React, { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export default function Home() {
  const [clientName, setClientName] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projectType, setProjectType] = useState("General");
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<{ type: "idle"|"loading"|"success"|"error"; message?: string }>({ type: "idle" });
  const [uploadedPath, setUploadedPath] = useState<string>("");
  const [availableFiles, setAvailableFiles] = useState<Array<{filename: string, type: string, size: number}>>([]);
  const [isPolling, setIsPolling] = useState(false);
  const [previousFileCount, setPreviousFileCount] = useState(0);

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

  async function fetchAvailableFiles() {
    try {
      const res = await fetch(`${API_BASE}/list_proposals`);
      if (res.ok) {
        const data = await res.json();
        const newFiles = data.files || [];
        
        // Check if new files were added
        if (newFiles.length > previousFileCount) {
          setStatus({ type: "success", message: "Proposal generated successfully!" });
        }
        
        setAvailableFiles(newFiles);
        setPreviousFileCount(newFiles.length);
      }
    } catch (err) {
      console.error("Failed to fetch files:", err);
    }
  }

  async function downloadFile(filename: string) {
    try {
      const response = await fetch(`${API_BASE}/download_proposal/${filename}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        throw new Error("Download failed");
      }
    } catch (err) {
      setStatus({ type: "error", message: `Failed to download ${filename}` });
    }
  }

  function getPdfForProposal(files: Array<{filename: string, type: string, size: number}>) {
    return files.find(f => f.type === '.pdf')?.filename || '';
  }

  function getDocxForProposal(files: Array<{filename: string, type: string, size: number}>) {
    return files.find(f => f.type === '.docx')?.filename || '';
  }

  function getProposalGroups() {
    const groups: { [key: string]: Array<{filename: string, type: string, size: number}> } = {};
    
    availableFiles.forEach(file => {
      // Extract timestamp from filename to group related files
      const match = file.filename.match(/(\d{8}_\d{6})/);
      if (match) {
        const timestamp = match[1];
        if (!groups[timestamp]) {
          groups[timestamp] = [];
        }
        groups[timestamp].push(file);
      } else {
        // Fallback grouping for files without timestamp
        const baseKey = file.filename.replace(/\.(pdf|docx|html|json)$/, '');
        if (!groups[baseKey]) {
          groups[baseKey] = [];
        }
        groups[baseKey].push(file);
      }
    });
    
    return Object.entries(groups).map(([key, files]) => ({ key, files }));
  }

  // Poll for files after successful generation
  React.useEffect(() => {
    if (isPolling) {
      const interval = setInterval(() => {
        fetchAvailableFiles();
        
        // Stop polling after finding new files
        if (availableFiles.length > previousFileCount) {
          setIsPolling(false);
        }
      }, 3000); // Poll every 3 seconds
      return () => clearInterval(interval);
    }
  }, [isPolling, availableFiles.length, previousFileCount]);

  // Fetch files on component mount
  React.useEffect(() => {
    fetchAvailableFiles();
  }, []);

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
      setStatus({ type: "loading", message: "Generating proposal... This may take a few minutes." });
      setIsPolling(true); // Start polling for generated files
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

        {availableFiles.length > 0 && (
          <section className="card" style={{ padding: 24, marginBottom: 16 }}>
            <div className="h2" style={{ marginBottom: 12 }}>Generated Proposals</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {getProposalGroups().map((group) => {
                const pdfFile = getPdfForProposal(group.files);
                const docxFile = getDocxForProposal(group.files);
                const htmlFile = group.files.find(f => f.type === '.html');
                const proposalName = htmlFile?.filename.replace('.html', '') || group.key;
                
                return (
                  <div key={group.key} style={{ border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden" }}>
                    <div style={{ padding: 16, background: "var(--surface)", borderBottom: "1px solid var(--border)" }}>
                      <div style={{ fontWeight: 600, color: "var(--color-black)", marginBottom: 8 }}>
                        {proposalName.replace(/proposal_/, '').replace(/_/g, ' ')}
                      </div>
                      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        {pdfFile && (
                          <button 
                            className="btn btn-secondary"
                            onClick={() => downloadFile(pdfFile)}
                            style={{ padding: "8px 12px", fontSize: 12 }}
                          >
                            ⬇️ Download PDF
                          </button>
                        )}
                        {docxFile && (
                          <button 
                            className="btn btn-secondary"
                            onClick={() => downloadFile(docxFile)}
                            style={{ padding: "8px 12px", fontSize: 12 }}
                          >
                            📝 Download DOCX
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        <section className="card" style={{ padding: 24 }}>
          <div className="h2" style={{ marginBottom: 8 }}>Tips</div>
          <ul className="muted" style={{ marginLeft: 18, lineHeight: 1.8 }}>
            <li>Use clear client and project names for organized outputs.</li>
            <li>Upload an RFP PDF to guide the generator with RAG.</li>
            <li>Files will appear here once generation is complete.</li>
          </ul>
        </section>
      </main>

      <footer className="app-footer">© {new Date().getFullYear()} Proposal Generator</footer>
    </div>
  );
}
