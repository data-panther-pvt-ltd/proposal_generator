import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import json
import os
from main import load_config, setup_logging, get_logger, get_all_agents, coordinator, process_rfp_pdf, SimpleCostTracker, ProposalGenerator

app = FastAPI(title="Proposal Generator API", version="2.0")

class ProposalRequestInput(BaseModel):
    pdf_path: str
    client_name: str
    project_name: str
    project_type: str


def run_proposal_task(input_data: ProposalRequestInput, config: dict):
    """Background task that generates the proposal and saves outputs"""
    logger = get_logger(__name__)

    async def _async_task():
        try:
            logger.info("=" * 60)
            logger.info("PROPOSAL GENERATOR v2.0-SDK - Starting via API (Background Task)")
            logger.info("=" * 60)

            # Initialize agents
            all_agents = get_all_agents()
            logger.info(f"✓ Available agents: {list(all_agents.keys())}")
            logger.info(f"✓ Coordinator agent: {coordinator.name}")

            # RFP config
            rfp_config = config.get("rfp", {})

            # Process RFP PDF (async)
            request = await process_rfp_pdf(
                input_data.pdf_path,
                input_data.client_name,
                input_data.project_name,
                input_data.project_type,
                rfp_config,
                config,
                logger
            )

            # Proposal generator (async)
            cost_tracker = SimpleCostTracker()
            generator = ProposalGenerator(cost_tracker=cost_tracker)

            proposal = await generator.generate_proposal(request)

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

            logger.info(f"✓ Proposal generation completed. Files saved: {json_file}, {html_file}")

        except Exception as e:
            logger.error(f"❌ Background proposal generation failed: {str(e)}")

    # Run async code inside sync background task
    asyncio.run(_async_task())


@app.post("/generate_proposal")
async def generate_proposal(input_data: ProposalRequestInput, background_tasks: BackgroundTasks):
    """FastAPI endpoint to queue proposal generation as background task"""

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

    # Schedule background task
    background_tasks.add_task(run_proposal_task, input_data, config)

    return {
        "status": "queued",
        "message": "Proposal generation started in background",
        "client_name": input_data.client_name,
        "project_name": input_data.project_name
    }
