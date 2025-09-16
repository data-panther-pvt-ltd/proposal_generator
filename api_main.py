import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import json
import os
from main import load_config, setup_logging, get_logger, get_all_agents, coordinator, create_request_from_rfp, SimpleCostTracker, ProposalGenerator
app = FastAPI(title="Proposal Generator API", version="2.0")

# In-memory status tracking
generation_status = {}

# Enable CORS for local development and typical frontend ports
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProposalRequestInput(BaseModel):
    pdf_path: str
    client_name: str
    project_name: str
    project_type: str


def run_proposal_task(input_data: ProposalRequestInput, config: dict, task_id: str):
    """Background task that generates the proposal and saves outputs"""
    logger = get_logger(__name__)

    async def _async_task():
        try:
            # Update status: starting
            generation_status[task_id] = {
                "status": "starting",
                "message": "Initializing proposal generation...",
                "timestamp": datetime.now().isoformat(),
                "pdf_generated": False
            }

            logger.info("=" * 60)
            logger.info("PROPOSAL GENERATOR v2.0-SDK - Starting via API (Background Task)")
            logger.info("=" * 60)

            # Initialize agents
            all_agents = get_all_agents()
            logger.info(f"✓ Available agents: {list(all_agents.keys())}")
            logger.info(f"✓ Coordinator agent: {coordinator.name}")

            # Update status: processing RFP
            generation_status[task_id] = {
                "status": "processing_rfp",
                "message": "Processing RFP document...",
                "timestamp": datetime.now().isoformat(),
                "pdf_generated": False
            }

            # RFP config
            rfp_config = config.get("rfp", {})

            # Process RFP PDF (async) using create_request_from_rfp
            request = await create_request_from_rfp(
                input_data.pdf_path,
                rfp_config,
                config,
                logger
            )

            # Override extracted values with user-provided inputs when present
            try:
                if input_data.client_name:
                    request.client_name = input_data.client_name
                if input_data.project_name:
                    request.project_name = input_data.project_name
                if input_data.project_type:
                    request.project_type = input_data.project_type
                if hasattr(request, 'requirements'):
                    request.requirements['source_pdf'] = input_data.pdf_path
            except Exception:
                pass

            # Update status: generating proposal
            generation_status[task_id] = {
                "status": "generating",
                "message": "Generating proposal content...",
                "timestamp": datetime.now().isoformat(),
                "pdf_generated": False
            }

            # Proposal generator (async)
            cost_tracker = SimpleCostTracker()
            generator = ProposalGenerator(cost_tracker=cost_tracker)

            proposal = await generator.generate_proposal(request)

            # Update status: saving files
            generation_status[task_id] = {
                "status": "saving",
                "message": "Saving proposal files...",
                "timestamp": datetime.now().isoformat(),
                "pdf_generated": bool(proposal.get('pdf_path'))
            }

            # Save outputs
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = Path(config["output"]["output_directory"])
            output_dir.mkdir(exist_ok=True)

            json_file = output_dir / f"proposal_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(proposal, f, indent=2, default=str)

            html_file = output_dir / f"proposal_{timestamp}.html"
            with open(html_file, "w") as f:
                f.write(proposal.get("html", ""))

            # Update final status
            pdf_generated = bool(proposal.get('pdf_path'))
            generation_status[task_id] = {
                "status": "completed",
                "message": f"✅ Proposal generation completed successfully! {('PDF generated successfully.' if pdf_generated else 'Note: PDF generation may have failed.')}",
                "timestamp": datetime.now().isoformat(),
                "pdf_generated": pdf_generated,
                "pdf_path": proposal.get('pdf_path', ''),
                "files": {
                    "json": str(json_file),
                    "html": str(html_file)
                }
            }

            logger.info(f"✓ Proposal generation completed. Files saved: {json_file}, {html_file}")
            if pdf_generated:
                logger.info(f"✅ PDF generated successfully: {proposal['pdf_path']}")

        except Exception as e:
            generation_status[task_id] = {
                "status": "failed",
                "message": f"❌ Proposal generation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "pdf_generated": False
            }
            logger.error(f"❌ Background proposal generation failed: {str(e)}")

    # Run async code inside sync background task
    asyncio.run(_async_task())


@app.post("/generate_proposal")
async def generate_proposal(input_data: ProposalRequestInput, background_tasks: BackgroundTasks):
    """FastAPI endpoint to queue proposal generation as background task"""
    import uuid

    # Load config
    config = load_config()
    setup_logging(config)
    logger = get_logger(__name__)

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )

    # Check if auto_process enabled
    rfp_config = config.get("rfp", {})
    if not rfp_config.get("auto_process", False):
        raise HTTPException(
            status_code=500,
            detail="RFP auto_process is not enabled in settings.yml"
        )

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Initialize status
    generation_status[task_id] = {
        "status": "queued",
        "message": "Proposal generation queued",
        "timestamp": datetime.now().isoformat(),
        "pdf_generated": False
    }

    # Schedule background task
    background_tasks.add_task(run_proposal_task, input_data, config, task_id)

    return {
        "status": "queued",
        "task_id": task_id,
        "message": "Proposal generation started in background",
        "client_name": input_data.client_name,
        "project_name": input_data.project_name
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Accept a PDF upload and save it to the uploads directory, returning the saved path."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = file.filename or "uploaded.pdf"
    destination = uploads_dir / safe_filename

    try:
        contents = await file.read()
        with open(destination, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    return {
        "status": "uploaded",
        "path": str(destination),
        "filename": safe_filename
    }


@app.get("/download_proposal/{filename}")
async def download_proposal(filename: str):
    """Download a generated proposal file (PDF, HTML, or JSON)"""
    # Security: only allow alphanumeric, dots, underscores, and hyphens
    import re
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Look for the file in the output directory
    output_dir = Path("generated_proposals")
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type based on extension
    if filename.endswith('.pdf'):
        media_type = 'application/pdf'
    elif filename.endswith('.html'):
        media_type = 'text/html'
    elif filename.endswith('.json'):
        media_type = 'application/json'
    elif filename.endswith('.docx'):
        media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    else:
        media_type = 'application/octet-stream'

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type
    )


@app.get("/list_proposals")
async def list_proposals():
    """List available generated proposal files"""
    output_dir = Path("generated_proposals")
    output_dir.mkdir(exist_ok=True)

    files = []
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime,
                "type": file_path.suffix
            })

    # Sort by modification time, newest first
    files.sort(key=lambda x: x["modified"], reverse=True)

    return {"files": files}


@app.get("/generation_status/{task_id}")
async def get_generation_status(task_id: str):
    """Get the status of a proposal generation task"""
    if task_id not in generation_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return generation_status[task_id]


@app.get("/generation_status")
async def get_all_generation_status():
    """Get all generation statuses (for debugging)"""
    return generation_status
