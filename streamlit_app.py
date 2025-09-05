import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd

from core.sdk_runner import ProposalRequest
from main import ProposalGenerator, load_config
from utils.logging_config import setup_logging, get_logger
from openai import OpenAI


def save_uploaded_file(uploaded_file, dest_path: Path) -> Optional[Path]:
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return dest_path
    except Exception:
        return None


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def main():
    st.set_page_config(page_title="AI Proposal Generator", layout="wide")
    st.title("AI Proposal Generator")
    st.caption("Upload RFP and your company info to generate a professional proposal.")

    config = load_config()
    setup_logging(config)
    logger = get_logger(__name__)

    base_dir = Path.cwd()
    uploads_dir = base_dir / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    with st.sidebar:
        st.header("Inputs")
        # OpenAI API Key (stored in session)
        if "OPENAI_API_KEY" not in st.session_state:
            st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state["OPENAI_API_KEY"],
            placeholder="sk-...",
            type="password",
            help="Your OpenAI API key will be kept only in this session.")
        st.session_state["OPENAI_API_KEY"] = api_key

        check_api = st.button("Check API Key", use_container_width=True)
        if check_api:
            if not st.session_state.get("OPENAI_API_KEY"):
                st.error("API key is empty. Please paste your OpenAI API key.")
            else:
                try:
                    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
                    client = OpenAI(api_key=st.session_state["OPENAI_API_KEY"])
                    # Lightweight connectivity check: list models
                    _ = client.models.list()
                    st.success("API key looks valid and OpenAI is reachable.")
                except Exception as e:
                    st.error(f"API key or connectivity check failed: {e}")

        rfp_file = st.file_uploader("RFP (PDF)", type=["pdf"], accept_multiple_files=False)
        profile_file = st.file_uploader("Company Profile (MD)", type=["md", "markdown"], accept_multiple_files=False)
        skills_internal = st.file_uploader("Skills - Internal (CSV)", type=["csv"], accept_multiple_files=False)
        skills_external = st.file_uploader("Skills - External (CSV)", type=["csv"], accept_multiple_files=False)

        st.divider()
        client_name = st.text_input("Client Name (optional)")
        project_name = st.text_input("Project Name (optional)")
        project_type = st.selectbox("Project Type", ["general", "it", "healthcare", "finance", "government", "other"], index=0)
        timeline = st.text_input("Timeline", value="As per RFP requirements")
        budget_range = st.text_input("Budget Range", value="As per RFP specifications")
        start_btn = st.button("Generate Proposal", type="primary", use_container_width=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Preview & Status")
        status_area = st.empty()
        html_area = st.empty()

    with col_right:
        st.subheader("Artifacts")
        artifacts_area = st.container()

    if start_btn:
        # Apply API key to environment for this run
        if st.session_state.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
        else:
            st.warning("OpenAI API key is empty; please provide it to proceed.")
            return

        if not rfp_file:
            st.error("Please upload an RFP PDF.")
            return

        # Save uploads
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rfp_dest = uploads_dir / f"rfp_{timestamp}.pdf"
        saved_rfp = save_uploaded_file(rfp_file, rfp_dest)
        if not saved_rfp:
            st.error("Failed to save RFP file.")
            return

        # Optionally save company profile and skills; update config at runtime
        if profile_file:
            profile_dest = uploads_dir / f"company_profile_{timestamp}.md"
            save_uploaded_file(profile_file, profile_dest)
            config['data']['company_profile'] = str(profile_dest)

        if skills_internal:
            skills_internal_dest = uploads_dir / f"skills_internal_{timestamp}.csv"
            save_uploaded_file(skills_internal, skills_internal_dest)
            config['data']['skills_internal'] = str(skills_internal_dest)

        if skills_external:
            skills_external_dest = uploads_dir / f"skills_external_{timestamp}.csv"
            save_uploaded_file(skills_external, skills_external_dest)
            config['data']['skills_external'] = str(skills_external_dest)

        # Override RFP path in config for this run
        config['rfp']['pdf_path'] = str(saved_rfp)

        status_area.info("Initializing generator...")
        try:
            generator = ProposalGenerator(cost_tracker=None)
            # Inject runtime config overrides if needed by downstream components
            generator.config.update(config)
        except Exception as e:
            st.error(f"Initialization failed: {e}")
            return

        # Build ProposalRequest similar to CLI flow
        from main import create_request_from_rfp
        try:
            status_area.info("Processing RFP and extracting info...")
            request = run_async(create_request_from_rfp(str(saved_rfp), config.get('rfp', {}), config, logger))
            if client_name:
                request.client_name = client_name
            if project_name:
                request.project_name = project_name
            request.project_type = project_type
            request.timeline = timeline
            request.budget_range = budget_range
        except Exception as e:
            st.error(f"RFP processing failed: {e}")
            return

        # Generate proposal
        try:
            status_area.info("Generating proposal with agents...")
            proposal = run_async(generator.generate_proposal(request))
        except Exception as e:
            st.error(f"Generation failed: {e}")
            return

        # Display HTML preview
        html = proposal.get('html')
        if html and len(html) > 0:
            with html_area:
                st.components.v1.html(html, height=900, scrolling=True)
        else:
            st.warning("No HTML content generated.")

        # Show artifacts and links
        with artifacts_area:
            st.write("Generated files:")
            pdf_path = proposal.get('pdf_path')
            docx_path = proposal.get('docx_path')
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, 'rb') as f:
                    st.download_button("Download PDF", f, file_name=Path(pdf_path).name, mime="application/pdf")
            if docx_path and Path(docx_path).exists():
                with open(docx_path, 'rb') as f:
                    st.download_button("Download DOCX", f, file_name=Path(docx_path).name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

            # Cost summary
            cost = generator.get_cost_summary()
            st.metric("Total Cost (USD)", f"${cost.get('total_cost', 0):.4f}")


if __name__ == "__main__":
    main()


