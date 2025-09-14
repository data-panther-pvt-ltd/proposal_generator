
"use client";
// import router, { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";
import { FiHome, FiFileText, FiSettings, FiLogOut, FiUpload, FiUser, FiBriefcase, FiClock, FiDollarSign, FiKey, FiCheck, FiX } from "react-icons/fi";
import { useRouter } from "next/navigation";

type GenerateResponse = {
  status?: string;
  message?: string;
  error?: string;
  client_name?: string;
  project_name?: string;
  task_id?: string;
};

type GenerationStatus = {
  status: string;
  message: string;
  timestamp: string;
  pdf_generated: boolean;
  pdf_path?: string;
  files?: {
    json: string;
    html: string;
  };
};

export default function Home() {
  const [rfpFile, setRfpFile] = useState<File | null>(null);
  const [profileFile, setProfileFile] = useState<File | null>(null);
  const [skillsInternalFile, setSkillsInternalFile] = useState<File | null>(null);
  const [skillsExternalFile, setSkillsExternalFile] = useState<File | null>(null);
  const [clientName, setClientName] = useState("");
  const [projectName, setProjectName] = useState("");
  const [projectType, setProjectType] = useState("general");
  const [timeline, setTimeline] = useState("As per RFP requirements");
  const [budgetRange, setBudgetRange] = useState("As per RFP specifications");
  const [apiKey, setApiKey] = useState("");
  const [apiKeyStoredAt, setApiKeyStoredAt] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [checkingKey, setCheckingKey] = useState(false);
  const [apiKeyValid, setApiKeyValid] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState("dashboard");
   const router = useRouter();

  const minutesLeft = useMemo(() => {
    if (!apiKeyStoredAt) return null;
    const elapsedMs = Date.now() - apiKeyStoredAt;
    const totalMs = 30 * 60 * 1000;
    const remaining = Math.max(0, totalMs - elapsedMs);
    return Math.ceil(remaining / 60000);
  }, [apiKeyStoredAt]);
  useEffect(() => {
    if (!apiKeyStoredAt) return;
    const interval = setInterval(() => {
      if (Date.now() - apiKeyStoredAt > 30 * 60 * 1000) {
        clearApiKey();
      }
    }, 10000);
    return () => clearInterval(interval);
  }, [apiKeyStoredAt]);

  function storeApiKeyLocally(value: string) {
    setApiKey(value);
    if (value) {
      setApiKeyStoredAt(Date.now());
      setApiKeyValid(null);
    }
  }

  function clearApiKey() {
    setApiKey("");
    setApiKeyStoredAt(null);
    setApiKeyValid(null);
  }

  const handleLogout = () => {
    localStorage.removeItem('auth');
    router.replace('/login');
  };

  async function handleCheckApiKey() {
    setCheckingKey(true);
    setErrorMessage(null);
    try {
      const res = await fetch("/api/check-openai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey }),
      });
      const data = await res.json();
      if (!res.ok || !data?.ok) {
        setApiKeyValid(false);
        throw new Error(data?.error || "API key check failed");
      }
      setApiKeyValid(true);
      setStatusMessage("API key verified successfully");
      if (!apiKeyStoredAt) setApiKeyStoredAt(Date.now());
    } catch (e: any) {
      setErrorMessage(e?.message || "Failed to verify API key");
    } finally {
      setCheckingKey(false);
    }
  }

//   async function handleSubmit() {
//     setErrorMessage(null);
//     setStatusMessage(null);
//     if (!rfpFile) {
//       setErrorMessage("Please select an RFP PDF file to continue.");
//       return;
//     }
//     try {
//       setIsSubmitting(true);
//       setStatusMessage("Processing RFP document...");
//       const form = new FormData();
//       form.append("file", rfpFile);
//       const uploadRes = await fetch("/api/upload", { method: "POST", body: form });
//       const uploadData = await uploadRes.json();
//       if (!uploadRes.ok) {
//         throw new Error(uploadData?.error || "Upload failed");
//       }
//       const savedPath: string = uploadData.savedPath;
//       let profilePath: string | null = null;
//       let skillsInternalPath: string | null = null;
//       let skillsExternalPath: string | null = null;
//       if (profileFile) {
//         const f = new FormData();
//         f.append("file", profileFile);
//         const r = await fetch("/api/upload", { method: "POST", body: f });
//         const d = await r.json();
//         if (!r.ok) throw new Error(d?.error || "Profile upload failed");
//         profilePath = d.savedPath;
//       }
//       if (skillsInternalFile) {
//         const f = new FormData();
//         f.append("file", skillsInternalFile);
//         const r = await fetch("/api/upload", { method: "POST", body: f });
//         const d = await r.json();
//         if (!r.ok) throw new Error(d?.error || "Internal skills upload failed");
//         skillsInternalPath = d.savedPath;
//       }
//       if (skillsExternalFile) {
//         const f = new FormData();
//         f.append("file", skillsExternalFile);
//         const r = await fetch("/api/upload", { method: "POST", body: f });
//         const d = await r.json();
//         if (!r.ok) throw new Error(d?.error || "External skills upload failed");
//         skillsExternalPath = d.savedPath;
//       }
//       setStatusMessage("Generating proposal...");
//       const genRes = await fetch("/api/generate", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({
//           pdf_path: savedPath,
//           client_name: clientName,
//           project_name: projectName,
//           project_type: projectType,
//           timeline,
//           budget_range: budgetRange,
//           profile_path: profilePath,
//           skills_internal_path: skillsInternalPath,
//           skills_external_path: skillsExternalPath,
//         }),
//       });
//       const genData: GenerateResponse = await genRes.json();
//       if (!genRes.ok) {
//         throw new Error(genData?.error || "Generation request failed");
//       }
//       setStatusMessage("Proposal generation completed successfully. Please check your output folder for the generated documents.");
//     } catch (err: any) {
//       setErrorMessage(err?.message || "An unexpected error occurred. Please try again.");
//     } finally {
//       setIsSubmitting(false);
//     }
//   }
const [generationCompleted, setGenerationCompleted] = useState(false);

async function handleSubmit() {
  setErrorMessage(null);
  setStatusMessage(null);
  setGenerationCompleted(false);

  if (!rfpFile) {
    setErrorMessage("Please select an RFP PDF file to continue.");
    return;
  }

  try {
    setIsSubmitting(true);
    setStatusMessage("Processing RFP document...");

    const form = new FormData();
    form.append("file", rfpFile);
    const uploadRes = await fetch("/api/upload", { method: "POST", body: form });
    const uploadData = await uploadRes.json();

    if (!uploadRes.ok) {
      throw new Error(uploadData?.error || "Upload failed");
    }

    const savedPath: string = uploadData.savedPath;
    let profilePath: string | null = null;
    let skillsInternalPath: string | null = null;
    let skillsExternalPath: string | null = null;

    if (profileFile) {
      const f = new FormData();
      f.append("file", profileFile);
      const r = await fetch("/api/upload", { method: "POST", body: f });
      const d = await r.json();
      if (!r.ok) throw new Error(d?.error || "Profile upload failed");
      profilePath = d.savedPath;
    }

    if (skillsInternalFile) {
      const f = new FormData();
      f.append("file", skillsInternalFile);
      const r = await fetch("/api/upload", { method: "POST", body: f });
      const d = await r.json();
      if (!r.ok) throw new Error(d?.error || "Internal skills upload failed");
      skillsInternalPath = d.savedPath;
    }

    if (skillsExternalFile) {
      const f = new FormData();
      f.append("file", skillsExternalFile);
      const r = await fetch("/api/upload", { method: "POST", body: f });
      const d = await r.json();
      if (!r.ok) throw new Error(d?.error || "External skills upload failed");
      skillsExternalPath = d.savedPath;
    }

    setStatusMessage("Generating proposal...");

    const genRes = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pdf_path: savedPath,
        client_name: clientName,
        project_name: projectName,
        project_type: projectType,
        timeline,
        budget_range: budgetRange,
        profile_path: profilePath,
        skills_internal_path: skillsInternalPath,
        skills_external_path: skillsExternalPath,
      }),
    });

    const genData: GenerateResponse = await genRes.json();

    if (!genRes.ok) {
      throw new Error(genData?.error || "Generation request failed");
    }

    // ✅ Poll for generation status including PDF generation
    setStatusMessage("Waiting for the proposal to be generated...");

    // Get task_id from response for status polling
    const taskId = genData.task_id;
    if (!taskId) {
      setErrorMessage("❌ No task ID returned from generation request");
      return;
    }

    const generationStatus = await pollForGenerationStatus(taskId, setStatusMessage);

    if (generationStatus) {
      setStatusMessage("✅ Proposal generated successfully. You can now download the files.");
      setGenerationCompleted(true);
    } else {
      setErrorMessage("⏱️ Proposal generation timed out. Please try again later.");
    }
  } catch (err: any) {
    setErrorMessage(err?.message || "An unexpected error occurred. Please try again.");
  } finally {
    setIsSubmitting(false);
  }
}




async function pollForGenerationStatus(
  taskId: string,
  statusSetter: (message: string) => void
): Promise<GenerationStatus | null> {
  const intervalMs = 3000; // Check every 3 seconds

  while (true) {
    try {
      const res = await fetch(`http://34.28.203.178:8003/generation_status/${taskId}`);
      const status: GenerationStatus = await res.json();

      if (res.ok) {
        // Update UI with current status message from API
        statusSetter(status.message || "Processing...");

        // Check if generation is completed (success or failure)
        if (status.status === "completed") {
          return status; // ✅ Generation completed
        } else if (status.status === "failed") {
          throw new Error(status.message || "Generation failed");
        }
        // Continue polling if still in progress
      }
    } catch (err) {
      if (err instanceof Error && err.message.includes("Generation failed")) {
        throw err;
      }
      console.error("Status polling error:", err);
    }

    await new Promise(res => setTimeout(res, intervalMs)); // Wait before retrying
  }
}







async function handleDownload(filename: string, extension: "pdf" | "docx") {
  try {
    // Optional: fetch list from backend and get latest filename
    const res = await fetch("http://34.28.203.178:8003/list_proposals");
    const data = await res.json();

    if (!res.ok || !data.files || !Array.isArray(data.files)) {
      throw new Error("Unable to fetch generated proposals");
    }

    // Find latest file with desired extension
    const filtered = data.files.filter((f: any) => f.type === `.${extension}`);
    if (filtered.length === 0) {
      alert(`No ${extension.toUpperCase()} proposals found.`);
      return;
    }

    // Sort by modified timestamp
    filtered.sort((a: any, b: any) => b.modified - a.modified);
    const latestFile = filtered[0].filename;

    // Trigger download
    const downloadUrl = `http://34.28.203.178:8003/download_proposal/${latestFile}`;
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = latestFile;
    a.click();
  } catch (err: any) {
    alert(`Failed to download ${extension.toUpperCase()} file: ${err.message}`);
  }
}


  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex flex-col">
      {/* Navbar */}
      <nav className="bg-white shadow-md py-4 px-6 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <img src='/proposallogo.png' alt="logo" className="h-8 w-"/>
        </div>
        
        {/* <div className="hidden md:flex items-center space-x-1 bg-gray-100 p-1 rounded-lg">
          <button 
            onClick={() => setActiveTab("dashboard")}
            className={`px-4 py-2 rounded-md flex items-center space-x-2 ${activeTab === "dashboard" ? "bg-white shadow" : "hover:bg-gray-200"}`}
          >
            <FiHome />
            <span>Dashboard</span>
          </button>
          <button 
            onClick={() => setActiveTab("proposals")}
            className={`px-4 py-2 rounded-md flex items-center space-x-2 ${activeTab === "proposals" ? "bg-white shadow" : "hover:bg-gray-200"}`}
          >
            <FiFileText />
            <span>Proposals</span>
          </button>
          <button 
            onClick={() => setActiveTab("settings")}
            className={`px-4 py-2 rounded-md flex items-center space-x-2 ${activeTab === "settings" ? "bg-white shadow" : "hover:bg-gray-200"}`}
          >
            <FiSettings />
            <span>Settings</span>
          </button>
        </div>
         */}
        <button 
          onClick={handleLogout}
          className="flex items-center space-x-2 text-red-600 hover:bg-red-100 px-2 py-1 rounded text-red-600 transition-colors duration-200"
        >
          <FiLogOut />
          <span>Logout</span>
        </button>
      </nav>

      {/* Main Content */}
      <main className="flex-grow py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-10 text-center">
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">AI Proposal Generator</h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Create professional proposals in minutes using AI-powered analysis of your RFP documents
            </p>
          </div>

          {/* Cards Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Required Documents Card */}
            <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
              <div className="flex items-center mb-5">
                <div className="w-10 h-10 rounded-lg bg-slate-200 flex items-center justify-center mr-3">
                  <FiFileText className="text-slate-600" />
                </div>
                <h2 className="text-xl font-Mobile text-gray-800">Required Documents</h2>
              </div>
              
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  RFP Document (PDF) *
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors duration-200">
                  <FiUpload className="mx-auto text-gray-400 text-2xl mb-2" />
                  <p className="text-sm text-gray-600 mb-2">
                    {rfpFile ? rfpFile.name : "Drag & drop or click to upload"}
                  </p>
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) => setRfpFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="rfp-upload"
                  />
                  <label htmlFor="rfp-upload" className="px-4 py-2 bg-blue-50 text-blue-600 rounded-lg text-sm font-medium hover:bg-blue-100 transition-colors duration-200 cursor-pointer">
                    Select File
                  </label>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Upload the Request for Proposal document in PDF format
                </p>
              </div>
            </div>

            {/* Project Information Card */}
            <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
              <div className="flex items-center mb-5">
                <div className="w-10 h-10 rounded-lg bg-slate-200 flex items-center justify-center mr-3">
                  <FiBriefcase className="text-slate-600" />
                </div>
                <h2 className="text-xl font-Mobile text-gray-800">Project Information</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Client Name
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <FiUser className="text-gray-400" />
                    </div>
                    <input
                      type="text"
                      value={clientName}
                      onChange={(e) => setClientName(e.target.value)}
                      placeholder="Enter client organization name"
                      className="w-full pl-10 pr-3 py-2 border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project Name
                  </label>
                  <input
                    type="text"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    placeholder="Enter project title"
                    className="w-full px-3 py-2 border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project Category
                  </label>
                  <select
                    value={projectType}
                    onChange={(e) => setProjectType(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white"
                  >
                    <option value="general">General</option>
                    <option value="it">Information Technology</option>
                    <option value="healthcare">Healthcare</option>
                    <option value="finance">Financial Services</option>
                    <option value="government">Government</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Supporting Documents Card */}
            <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
              <div className="flex items-center mb-5">
                <div className="w-10 h-10 rounded-lg bg-slate-200 flex items-center justify-center mr-3">
                  <FiUpload className="text-slate-600" />
                </div>
                <h2 className="text-xl font-Mobile text-gray-800">Supporting Documents</h2>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Company Profile (Markdown)
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-green-400 transition-colors duration-200">
                    <p className="text-sm text-gray-600 mb-2">
                      {profileFile ? profileFile.name : "No file selected"}
                    </p>
                    <input
                      type="file"
                      accept=".md,.markdown,text/markdown"
                      onChange={(e) => setProfileFile(e.target.files?.[0] || null)}
                      className="hidden"
                      id="profile-upload"
                    />
                    <label htmlFor="profile-upload" className="px-3 py-1 bg-green-50 text-green-600 rounded-lg text-sm font-medium hover:bg-green-100 transition-colors duration-200 cursor-pointer">
                      Select File
                    </label>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Internal Skills Database (CSV)
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-green-400 transition-colors duration-200">
                    <p className="text-sm text-gray-600 mb-2">
                      {skillsInternalFile ? skillsInternalFile.name : "No file selected"}
                    </p>
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(e) => setSkillsInternalFile(e.target.files?.[0] || null)}
                      className="hidden"
                      id="skills-internal-upload"
                    />
                    <label htmlFor="skills-internal-upload" className="px-3 py-1 bg-green-50 text-green-600 rounded-lg text-sm font-medium hover:bg-green-100 transition-colors duration-200 cursor-pointer">
                      Select File
                    </label>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    External Skills Database (CSV)
                  </label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-green-400 transition-colors duration-200">
                    <p className="text-sm text-gray-600 mb-2">
                      {skillsExternalFile ? skillsExternalFile.name : "No file selected"}
                    </p>
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={(e) => setSkillsExternalFile(e.target.files?.[0] || null)}
                      className="hidden"
                      id="skills-external-upload"
                    />
                    <label htmlFor="skills-external-upload" className="px-3 py-1 bg-green-50 text-green-600 rounded-lg text-sm font-medium hover:bg-green-100 transition-colors duration-200 cursor-pointer">
                      Select File
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* API Configuration Card */}
            <div className="bg-white rounded-xl shadow-md p-6 border border-gray-100">
              <div className="flex items-center mb-5">
                <div className="w-10 h-10 rounded-lg bg-slate-200 flex items-center justify-center mr-3">
                  <FiKey className="text-slate-600" />
                </div>
                <h2 className="text-xl font-Mobile text-gray-800">AI Configuration</h2>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  OpenAI API Key *
                </label>
                <div className="flex gap-2 mb-2">
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(e) => storeApiKeyLocally(e.target.value)}
                    placeholder="sk-..."
                    className="flex-1 px-3 py-2 border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <button
                    type="button"
                    onClick={clearApiKey}
                    className="px-4 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-700 font-medium hover:bg-gray-100"
                  >
                    Clear
                  </button>
                  <button
                    type="button"
                    disabled={!apiKey || checkingKey}
                    onClick={handleCheckApiKey}
                    className="px-4 py-2 bg-slate-950 text-white rounded-lg font-medium hover:bg-purple-700 disabled:bg-red-400 disabled:cursor-not-allowed flex items-center"
                  >
                    {checkingKey ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Verifying...
                      </>
                    ) : "Verify"}
                  </button>
                </div>
                
                {apiKeyStoredAt && (
                  <div className="flex items-center gap-3 mb-2 text-sm text-gray-600">
                    <span>Auto-clear in {minutesLeft ?? 0} minutes</span>
                    {apiKeyValid === true && (
                      <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium flex items-center">
                        <FiCheck className="mr-1" /> Verified
                      </span>
                    )}
                    {apiKeyValid === false && (
                      <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs font-medium flex items-center">
                        <FiX className="mr-1" /> Invalid
                      </span>
                    )}
                  </div>
                )}
                
                <p className="text-xs text-gray-500">
                  Your API key is stored temporarily and will be cleared automatically for security
                </p>
              </div>
              
              <div className="mt-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project Timeline
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <FiClock className="text-gray-400" />
                    </div>
                    <input
                      type="text"
                      value={timeline}
                      onChange={(e) => setTimeline(e.target.value)}
                      className="w-full pl-10 pr-3 py-2 border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Budget Range
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <FiDollarSign className="text-gray-400" />
                    </div>
                    <input
                      type="text"
                      value={budgetRange}
                      onChange={(e) => setBudgetRange(e.target.value)}
                      className="w-full pl-10 pr-3 py-2 border border-gray-300 text-gray-900 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Generate Button */}
          <div className="flex justify-center">
            <button
              type="button"
              disabled={isSubmitting || !rfpFile}
              onClick={handleSubmit}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-xl font-Mobile text-lg hover:from-blue-700 hover:to-indigo-800 disabled:from-green-400 disabled:to-green-400 disabled:cursor-not-allowed transition-all duration-300 transform hover:-translate-y-1 hover:shadow-xl flex items-center min-w-[250px] justify-center"
            >
              {isSubmitting ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                "Generate Proposal"
              )}
            </button>
          </div>

          {/* Status Messages */}
          {(statusMessage || errorMessage) && (
            <div className="mt-8 space-y-3">
              {statusMessage && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-blue-800 font-medium flex items-start">
                  <FiCheck className="mt-0.5 mr-2 flex-shrink-0" />
                  <span>{statusMessage}</span>
                </div>
              )}
              {errorMessage && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-800 font-medium flex items-start">
                  <FiX className="mt-0.5 mr-2 flex-shrink-0" />
                  <span>{errorMessage}</span>
                </div>
              )}
            </div>
          )}


      {/* Download Buttons */}
<div className="mt-8 flex justify-center gap-4">
  <button
    onClick={() => handleDownload("latest", "pdf")}
    className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
  >
    Download PDF
  </button>
  <button
    onClick={() => handleDownload("latest", "docx")}
    className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
  >
    Download DOCX
  </button>
</div>

        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-6 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center text-sm text-gray-600">
          <p>Ensure your backend service is running with proper OpenAI API configuration. Generated proposals will be saved according to your system settings.</p>
          <p className="mt-2">© {new Date().getFullYear()} Data Panther. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}