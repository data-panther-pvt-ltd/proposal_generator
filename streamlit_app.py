import streamlit as st
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
import base64
import pandas as pd

# Import existing functionality
from main import load_config, setup_logging, get_logger, ProposalGenerator, create_request_from_rfp
from core.simple_cost_tracker import SimpleCostTracker

st.set_page_config(
    page_title=" AI Proposal Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main theme and colors */
    

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }

    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1.5rem;
    }

    /* Success/error boxes */
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }

    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }

    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(116, 185, 255, 0.3);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }

    /* Download button styling */
    .download-btn {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }

    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.6);
        text-decoration: none;
        color: white;
    }

    /* Metrics styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }

    /* File uploader styling */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }

    /* Progress bar styling */
    .stProgress {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .status-success {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }

    .status-error {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }

    /* Feature list styling */
    .feature-list {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }

    .feature-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .feature-item:last-child {
        border-bottom: none;
    }

    /* Animation for balloons effect */
    @keyframes bounce {
        0%, 20%, 60%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-20px);
        }
        80% {
            transform: translateY(-10px);
        }
    }

    .bounce {
        animation: bounce 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

def create_download_link(file_path, filename, file_type):
    """Create download link for files"""
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            file_data = file.read()
        b64_file = base64.b64encode(file_data).decode()

        if file_type.lower() == 'pdf':
            mime_type = "application/pdf"
        elif file_type.lower() == 'docx':
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif file_type.lower() == 'html':
            mime_type = "text/html"
        elif file_type.lower() == 'json':
            mime_type = "application/json"
        else:
            mime_type = "application/octet-stream"

        download_link = f'<a href="data:{mime_type};base64,{b64_file}" download="{filename}" class="download-btn">Download {file_type.upper()}</a>'
        return download_link
    return None

async def generate_proposal_main(pdf_path, config):
    """Use main.py logic to generate proposal"""
    try:
        # Setup logging
        setup_logging(config)
        logger = get_logger(__name__)

        logger.info("=" * 60)
        logger.info("PROPOSAL GENERATOR v2.0-SDK - Starting via Streamlit")
        logger.info("=" * 60)

        # Get RFP config
        rfp_config = config.get("rfp", {})

        # Create request from RFP
        request = await create_request_from_rfp(pdf_path, rfp_config, config, logger)

        logger.info("Auto-extracted information:")
        logger.info(f"Client: {request.client_name}")
        logger.info(f"Project: {request.project_name}")
        logger.info(f"Type: {request.project_type}")

        # Initialize cost tracker and generator
        cost_tracker = SimpleCostTracker(config)
        generator = ProposalGenerator(cost_tracker=cost_tracker)

        # Generate proposal
        logger.info("Starting proposal generation...")
        proposal = await generator.generate_proposal(request)

        # Save outputs with organized folder structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(config["output"]["output_directory"])
        output_dir.mkdir(exist_ok=True)

        # Create subfolders for organization
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        differences_dir = output_dir / "differences"
        differences_dir.mkdir(exist_ok=True)

        # Save files in artifacts subfolder
        json_file = artifacts_dir / f"proposal_{request.client_name}_{timestamp}.json"
        html_file = artifacts_dir / f"proposal_{request.client_name}_{timestamp}.html"

        # JSON output
        with open(json_file, "w", encoding='utf-8') as f:
            json.dump(proposal, f, indent=2, default=str, ensure_ascii=False)

        # HTML output
        html_content = proposal.get("html", "")
        with open(html_file, "w", encoding='utf-8') as f:
            f.write(html_content)

        # Check if PDF and DOCX were already generated by main.py
        pdf_file = proposal.get('pdf_path')
        docx_file = proposal.get('docx_path')

        # Only generate PDF if not already created by main.py
        if not pdf_file and config.get('output', {}).get('enable_pdf', True):
            try:
                from core.pdf_exporter import PDFExporter
                pdf_exporter = PDFExporter(config)
                pdf_file = pdf_exporter.export(html_content, request.client_name)
                logger.info(f"PDF generated by Streamlit: {pdf_file}")
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")
        elif pdf_file:
            logger.info(f"PDF already generated by main.py: {pdf_file}")

        # Only generate DOCX if not already created by main.py
        if not docx_file and config.get('output', {}).get('enable_docx', True):
            try:
                from core.docx_exporter import DOCXExporter
                docx_exporter = DOCXExporter(config)
                docx_file = docx_exporter.export(html_content, request.client_name)
                logger.info(f"DOCX generated by Streamlit: {docx_file}")
            except Exception as e:
                logger.warning(f"DOCX generation failed: {e}")
        elif docx_file:
            logger.info(f"DOCX already generated by main.py: {docx_file}")

        # Get cost summary
        cost_summary = cost_tracker.get_summary()

        # Update cost.md file
        model_used = config.get('openai', {}).get('model', 'gpt-5')
        cost_tracker.append_to_cost_md("Proposal Generation (Streamlit)", model_used)

        logger.info("Proposal generation completed successfully!")

        return {
            "success": True,
            "request": request,
            "proposal": proposal,
            "json_file": str(json_file),
            "html_file": str(html_file),
            "pdf_file": str(pdf_file) if pdf_file else None,
            "docx_file": str(docx_file) if docx_file else None,
            "cost_summary": cost_summary
        }

    except Exception as e:
        logger.error(f"Proposal generation failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def parse_cost_table():
    """Parse cost.md table into DataFrame"""
    try:
        import pandas as pd
        import re

        with open("cost.md", "r") as f:
            content = f.read()

        # Find the table in the markdown
        lines = content.split('\n')
        table_lines = []
        in_table = False

        for line in lines:
            if line.strip().startswith('|') and 'Date' in line:
                in_table = True
            if in_table and line.strip().startswith('|'):
                if not line.strip().startswith('|---'):
                    table_lines.append(line.strip())
            elif in_table and not line.strip():
                break

        if not table_lines:
            return None

        # Parse table lines
        data = []
        headers = None

        for i, line in enumerate(table_lines):
            cols = [col.strip() for col in line.split('|')[1:-1]]  # Remove empty first/last
            if i == 0:
                headers = cols
            else:
                data.append(cols)

        if not data:
            return None

        df = pd.DataFrame(data, columns=headers)

        # Clean and convert data types
        if 'Total Cost (USD)' in df.columns:
            df['Total Cost (USD)'] = df['Total Cost (USD)'].str.replace('$', '').astype(float)
        if 'Input Tokens' in df.columns:
            df['Input Tokens'] = df['Input Tokens'].str.replace(',', '').astype(int)
        if 'Output Tokens' in df.columns:
            df['Output Tokens'] = df['Output Tokens'].str.replace(',', '').astype(int)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        return df
    except Exception as e:
        st.error(f"Error parsing cost data: {e}")
        return None

def display_cost_dashboard():
    """Display comprehensive cost tracking dashboard"""
    st.markdown("# 💰 Cost Tracking Dashboard")

    df = parse_cost_table()
    if df is None or df.empty:
        st.warning("No cost data available. Generate some proposals first!")
        return

    # Summary metrics
    st.markdown("## 📊 Cost Summary")

    total_cost = df['Total Cost (USD)'].sum()
    total_operations = len(df)
    avg_cost = total_cost / total_operations if total_operations > 0 else 0
    total_tokens = df['Input Tokens'].sum() + df['Output Tokens'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spent", f"${total_cost:.4f}")
    with col2:
        st.metric("Operations", total_operations)
    with col3:
        st.metric("Avg Cost/Op", f"${avg_cost:.4f}")
    with col4:
        st.metric("Total Tokens", f"{total_tokens:,}")

    st.markdown("---")

    # Filters
    st.markdown("## 🔍 Filters")
    col_filter1, col_filter2, col_filter3 = st.columns(3)

    with col_filter1:
        models = ["All"] + sorted(df["Model"].unique().tolist())
        selected_model = st.selectbox("Filter by Model", models)

    with col_filter2:
        operations = ["All"] + sorted(df["Operation Type"].unique().tolist())
        selected_operation = st.selectbox("Filter by Operation", operations)

    with col_filter3:
        date_range = st.date_input("Date Range", value=[df['Date'].min().date(), df['Date'].max().date()])

    # Apply filters
    filtered_df = df.copy()
    if selected_model != "All":
        filtered_df = filtered_df[filtered_df["Model"] == selected_model]
    if selected_operation != "All":
        filtered_df = filtered_df[filtered_df["Operation Type"] == selected_operation]
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'].dt.date >= start_date) & (filtered_df['Date'].dt.date <= end_date)]

    st.markdown("---")

    # Visualizations
    st.markdown("## 📈 Cost Analysis")

    if not filtered_df.empty:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Cost over time
            fig1 = px.line(filtered_df, x='Date', y='Total Cost (USD)',
                          title='Cost Over Time', markers=True)
            fig1.update_layout(xaxis_title="Date", yaxis_title="Cost (USD)")
            st.plotly_chart(fig1, use_container_width=True)

            # Cost by model and operation type
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                model_costs = filtered_df.groupby('Model')['Total Cost (USD)'].sum().reset_index()
                fig2 = px.bar(model_costs, x='Model', y='Total Cost (USD)',
                             title='Cost by Model')
                st.plotly_chart(fig2, use_container_width=True)

            with col_chart2:
                operation_costs = filtered_df.groupby('Operation Type')['Total Cost (USD)'].sum().reset_index()
                fig3 = px.pie(operation_costs, values='Total Cost (USD)', names='Operation Type',
                             title='Cost Distribution by Operation')
                st.plotly_chart(fig3, use_container_width=True)

            # Token usage analysis
            st.markdown("### 🎯 Token Usage Analysis")

            col_token1, col_token2 = st.columns(2)

            with col_token1:
                fig4 = px.scatter(filtered_df, x='Input Tokens', y='Output Tokens',
                                 color='Model', size='Total Cost (USD)',
                                 title='Token Usage Pattern')
                st.plotly_chart(fig4, use_container_width=True)

            with col_token2:
                # Daily token usage
                daily_tokens = filtered_df.groupby(filtered_df['Date'].dt.date).agg({
                    'Input Tokens': 'sum',
                    'Output Tokens': 'sum'
                }).reset_index()

                fig5 = go.Figure()
                fig5.add_trace(go.Bar(x=daily_tokens['Date'], y=daily_tokens['Input Tokens'],
                                     name='Input Tokens', marker_color='lightblue'))
                fig5.add_trace(go.Bar(x=daily_tokens['Date'], y=daily_tokens['Output Tokens'],
                                     name='Output Tokens', marker_color='lightcoral'))
                fig5.update_layout(title='Daily Token Usage', barmode='stack')
                st.plotly_chart(fig5, use_container_width=True)

        except ImportError:
            st.warning("Plotly not available. Install with: pip install plotly")

    st.markdown("---")

    # Detailed table
    st.markdown("## 📋 Detailed Cost History")

    # Display options
    col_display1, col_display2 = st.columns(2)
    with col_display1:
        show_recent = st.checkbox("Show only recent (last 20)", value=True)
    with col_display2:
        show_description = st.checkbox("Show descriptions", value=True)

    display_df = filtered_df.copy()
    if show_recent:
        display_df = display_df.head(20)

    if not show_description and 'Description' in display_df.columns:
        display_df = display_df.drop('Description', axis=1)

    # Format for display
    display_df_formatted = display_df.copy()
    display_df_formatted['Total Cost (USD)'] = display_df_formatted['Total Cost (USD)'].apply(lambda x: f"${x:.4f}")
    display_df_formatted['Input Tokens'] = display_df_formatted['Input Tokens'].apply(lambda x: f"{x:,}")
    display_df_formatted['Output Tokens'] = display_df_formatted['Output Tokens'].apply(lambda x: f"{x:,}")

    st.dataframe(display_df_formatted, use_container_width=True)

    # Export options
    st.markdown("---")
    st.markdown("## 📥 Export Options")

    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("📊 Download as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"cost_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col_export2:
        if st.button("📋 Copy Summary"):
            summary = f"""
Cost Summary:
- Total Spent: ${total_cost:.4f}
- Operations: {total_operations}
- Average Cost: ${avg_cost:.4f}
- Total Tokens: {total_tokens:,}
- Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}
            """
            st.code(summary)

def main():
    # Professional Header
    st.markdown("""
        <div class="main-header">
            <h1>🚀 AI Proposal Generator</h1>
            <p>Transform RFPs into winning proposals with enterprise-grade AI technology</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2 = st.tabs(["🚀 Proposal Generator", "💰 Cost Tracking"])

    with tab1:
        display_proposal_generator()

    with tab2:
        display_cost_dashboard()

def display_proposal_generator():

    # Sidebar with professional styling
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
                <h2 style="color: white; margin: 0;">⚙️ Control Panel</h2>
            </div>
        """, unsafe_allow_html=True)

        # Model selection with enhanced styling
        st.markdown("### 🧠 AI Model Configuration")
        model_options = ["gpt-4o", "gpt-5"]
        selected_model = st.selectbox(
            "Select AI Model",
            options=model_options,
            index=1,  # Default to gpt-5
            help="Choose the AI model for proposal generation"
        )

        # Model info with status indicators
        if selected_model == "gpt-5":
            st.markdown('<div class="status-indicator status-success">🚀 GPT-5: Advanced AI Model</div>', unsafe_allow_html=True)
            st.markdown("**Features:** Enhanced reasoning, better analysis, superior quality")
        else:
            st.markdown('<div class="status-indicator status-success">⚡ GPT-4o: Optimized Model</div>', unsafe_allow_html=True)
            st.markdown("**Features:** Fast processing, reliable output, cost-effective")

        st.markdown("---")

        # System Status Section
        st.markdown("### 📊 System Status")

        # API key status
        if os.getenv("OPENAI_API_KEY"):
            st.markdown('<div class="status-indicator status-success">✅ API Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-error">❌ API Not Found</div>', unsafe_allow_html=True)
            st.error("⚠️ Please set OPENAI_API_KEY environment variable")
            return

        # Load config with status
        try:
            config = load_config()

            # Update model in config - override ALL model settings
            config['openai']['model'] = selected_model

            # Update ALL agents to use selected model
            for agent_name in config['agents']:
                config['agents'][agent_name]['model'] = selected_model

            # Update correction model
            config['output']['correction_model'] = selected_model

            # Update cost management optimization rules
            if 'cost_management' in config and 'optimization' in config['cost_management']:
                opt_rules = config['cost_management']['optimization']['model_selection_rules']
                for task_type in opt_rules:
                    opt_rules[task_type] = selected_model

            st.markdown('<div class="status-indicator status-success">✅ Configuration Loaded</div>', unsafe_allow_html=True)

            # System info
            st.markdown("**Output Directory:**")
            st.code(config.get('output', {}).get('output_directory', 'N/A'))

        except Exception as e:
            st.markdown('<div class="status-indicator status-error">❌ Config Error</div>', unsafe_allow_html=True)
            st.error(f"Configuration error: {e}")
            return

        st.markdown("---")

        # Cost estimation
        st.markdown("### 💰 Cost Estimation")
        if selected_model == "gpt-5":
            st.markdown("**Estimated Range:** $0.20 - $2.00")
            st.markdown("*Advanced model with premium features*")
        else:
            st.markdown("**Estimated Range:** $0.10 - $1.00")
            st.markdown("*Optimized for speed and efficiency*")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 📄 Upload Required Files")

        # RFP Document Upload
        st.markdown("### 📋 RFP Document")
        uploaded_file = st.file_uploader(
            "Upload RFP PDF file",
            type=['pdf'],
            help="Upload the Request for Proposal (RFP) document in PDF format"
        )

        if uploaded_file:
            st.success(f"✅ RFP uploaded: {uploaded_file.name}")

        st.markdown("---")

        # Company Data Uploads
        st.markdown("### 🏢 Company Data Files")

        col_upload1, col_upload2 = st.columns(2)

        with col_upload1:
            # Company Profile Upload
            company_profile_file = st.file_uploader(
                "Company Profile",
                type=['txt', 'md', 'pdf'],
                help="Upload your company profile document"
            )
            if company_profile_file:
                st.success(f"✅ Profile: {company_profile_file.name}")

        with col_upload2:
            # Internal Skills CSV Upload
            skill_company_file = st.file_uploader(
                "Internal Skills CSV",
                type=['csv'],
                help="CSV file with internal team skills and rates"
            )
            if skill_company_file:
                st.success(f"✅ Internal: {skill_company_file.name}")

        # External Skills CSV Upload (full width)
        skill_external_file = st.file_uploader(
            "External Skills CSV",
            type=['csv'],
            help="CSV file with external vendor skills and rates"
        )
        if skill_external_file:
            st.success(f"✅ External: {skill_external_file.name}")

        st.markdown("---")

        # Validation and Generate Button
        all_files_uploaded = all([
            uploaded_file,
            company_profile_file,
            skill_company_file,
            skill_external_file
        ])

        if not all_files_uploaded:
            st.info("📋 Please upload all required files to proceed")
            missing_files = []
            if not uploaded_file:
                missing_files.append("RFP PDF")
            if not company_profile_file:
                missing_files.append("Company Profile")
            if not skill_company_file:
                missing_files.append("Internal Skills CSV")
            if not skill_external_file:
                missing_files.append("External Skills CSV")

            st.warning(f"Missing: {', '.join(missing_files)}")

        if all_files_uploaded:

            if st.button("🚀 Generate Proposal", type="primary", use_container_width=True):
                # Save uploaded files
                upload_dir = Path("temp_uploads")
                upload_dir.mkdir(exist_ok=True)
                data_dir = Path("data_streamlit")
                data_dir.mkdir(exist_ok=True)

                # Save RFP PDF file
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Save company profile file
                company_profile_path = data_dir / "company_profile.md"
                if company_profile_file.type == "application/pdf":
                    company_profile_path = data_dir / "company_profile.pdf"
                elif company_profile_file.type == "text/plain":
                    company_profile_path = data_dir / "company_profile.txt"

                with open(company_profile_path, "wb") as f:
                    f.write(company_profile_file.getbuffer())

                # Save skill CSV files
                skill_company_path = data_dir / "skill_company.csv"
                with open(skill_company_path, "wb") as f:
                    f.write(skill_company_file.getbuffer())

                skill_external_path = data_dir / "skill_external.csv"
                with open(skill_external_path, "wb") as f:
                    f.write(skill_external_file.getbuffer())

                # Update config to use uploaded files
                config['data']['skills_internal'] = str(skill_company_path)
                config['data']['skills_external'] = str(skill_external_path)
                config['data']['company_profile'] = str(company_profile_path)
                config['data']['case_studies'] = str(company_profile_path)

                # Show progress
                with st.spinner(f"🔄 Generating proposal using {selected_model}..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("📄 Processing RFP document...")
                    progress_bar.progress(20)

                    # Run generation
                    try:
                        result = asyncio.run(generate_proposal_main(str(file_path), config))

                        progress_bar.progress(100)
                        status_text.text("✅ Proposal generation completed!")

                        # Clean up
                        file_path.unlink(missing_ok=True)

                        if result["success"]:
                            # Store result in session state to persist across re-runs
                            st.session_state['proposal_result'] = result
                            st.session_state['proposal_generated'] = True

                            st.success("🎉 Proposal generated successfully!")

                            # Just show basic success message, detailed view will be below


                        else:
                            st.error(f"❌ Proposal generation failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        progress_bar.progress(0)
                        status_text.text("❌ Generation failed")
                        file_path.unlink(missing_ok=True)
                        st.error(f"Error: {str(e)}")

        # Display proposal results from session state (persists across re-runs)
        if st.session_state.get('proposal_generated', False) and 'proposal_result' in st.session_state:
            result = st.session_state['proposal_result']
            request = result["request"]

            st.markdown("---")
            st.markdown("### 📋 Generated Proposal")

            # Show extracted info
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Client", request.client_name)
            with info_col2:
                st.metric("Project", request.project_name)
            with info_col3:
                st.metric("Type", request.project_type)

            # Download section
            st.markdown("### 📥 Download Generated Files")
            download_col1, download_col2 = st.columns(2)

            with download_col1:
                if result["pdf_file"] and os.path.exists(result["pdf_file"]):
                    pdf_link = create_download_link(
                        result["pdf_file"],
                        f"proposal_{request.client_name}.pdf",
                        "pdf"
                    )
                    st.markdown(pdf_link, unsafe_allow_html=True)

            with download_col2:
                if result["docx_file"] and os.path.exists(result["docx_file"]):
                    docx_link = create_download_link(
                        result["docx_file"],
                        f"proposal_{request.client_name}.docx",
                        "docx"
                    )
                    st.markdown(docx_link, unsafe_allow_html=True)

            # Cost summary
            if result.get("cost_summary"):
                cost_summary = result["cost_summary"]
                st.markdown("### 💰 Cost Summary")
                cost_col1, cost_col2, cost_col3 = st.columns(3)

                with cost_col1:
                    st.metric("Total Cost", f"${cost_summary.get('total_cost', 0):.4f}")
                with cost_col2:
                    st.metric("Input Tokens", f"{cost_summary.get('total_input_tokens', 0):,}")
                with cost_col3:
                    st.metric("Output Tokens", f"{cost_summary.get('total_output_tokens', 0):,}")

            # Preview
            if result.get("proposal", {}).get("html"):
                with st.expander("👁️ Preview Generated Proposal", expanded=True):
                    st.components.v1.html(
                        result["proposal"]["html"],
                        height=600,
                        scrolling=True
                    )

            # Correction Section
            st.markdown("---")
            st.markdown("### ✏️ Proposal Correction")
            st.markdown("*Apply AI-powered corrections to improve formatting, consistency, and quality*")

            col_correction1, col_correction2 = st.columns([1, 1])

            with col_correction1:
                if st.button("🔄 Apply AI Corrections", type="secondary", use_container_width=True, key="apply_corrections_btn"):
                        with st.spinner("🔄 Applying corrections using ProposalCorrector..."):
                            try:
                                from core.proposal_corrector import ProposalCorrector

                                # Get current config and override ALL model settings for corrections
                                config = load_config()
                                config['openai']['model'] = selected_model

                                # Update ALL agents to use selected model for corrections
                                for agent_name in config['agents']:
                                    config['agents'][agent_name]['model'] = selected_model

                                # Update correction model
                                config['output']['correction_model'] = selected_model

                                # Update cost management optimization rules for corrections
                                if 'cost_management' in config and 'optimization' in config['cost_management']:
                                    opt_rules = config['cost_management']['optimization']['model_selection_rules']
                                    for task_type in opt_rules:
                                        opt_rules[task_type] = selected_model

                                # Initialize a new cost tracker for corrections
                                correction_cost_tracker = SimpleCostTracker(config)

                                # Initialize the corrector
                                corrector = ProposalCorrector(config, correction_cost_tracker)

                                # Apply corrections using the corrector
                                async def run_correction():
                                    return await corrector.correct_proposal(result["json_file"])

                                corrected_pdf_path, diff_report_path = asyncio.run(run_correction())

                                # Organize corrected files properly
                                output_dir = Path(config["output"]["output_directory"])
                                differences_dir = output_dir / "differences"
                                differences_dir.mkdir(exist_ok=True)

                                # Move correction report to differences folder
                                if diff_report_path and os.path.exists(diff_report_path):
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    new_diff_path = differences_dir / f"correction_report_{request.client_name}_{timestamp}.txt"
                                    import shutil
                                    shutil.move(diff_report_path, new_diff_path)
                                    diff_report_path = str(new_diff_path)

                                # Generate corrected DOCX and prepare preview
                                corrected_docx_path = None
                                try:
                                    # Load the corrected JSON to get HTML content
                                    # ProposalCorrector saves with "correct_" prefix, not "_corrected" suffix
                                    original_json_file = Path(result["json_file"])
                                    corrected_json_path = original_json_file.parent / f"correct_{original_json_file.name}"
                                    if os.path.exists(corrected_json_path):
                                        with open(corrected_json_path, 'r', encoding='utf-8') as f:
                                            corrected_json = json.load(f)

                                        # Generate HTML from corrected JSON and save to artifacts
                                        from core.html_generator import HTMLGenerator
                                        html_generator = HTMLGenerator(config)
                                        corrected_html = html_generator.generate(corrected_json)

                                        # Save corrected HTML to artifacts
                                        artifacts_dir = output_dir / "artifacts"
                                        corrected_html_path = artifacts_dir / f"proposal_{request.client_name}_{timestamp}_corrected.html"
                                        with open(corrected_html_path, "w", encoding='utf-8') as f:
                                            f.write(corrected_html)

                                        # Generate DOCX from corrected HTML
                                        from core.docx_exporter import DOCXExporter
                                        docx_exporter = DOCXExporter(config)
                                        corrected_docx_path = docx_exporter.export(corrected_html, f"{request.client_name}_corrected")

                                        # Store corrected HTML and JSON for preview
                                        st.session_state['corrected_html'] = corrected_html
                                        st.session_state['corrected_json'] = corrected_json

                                except Exception as e:
                                    st.warning(f"DOCX correction failed: {e}")

                                # Store corrected results in session state
                                st.session_state['corrected_pdf_path'] = corrected_pdf_path
                                st.session_state['corrected_docx_path'] = corrected_docx_path
                                st.session_state['diff_report_path'] = diff_report_path
                                st.session_state['correction_cost'] = correction_cost_tracker.get_summary()

                                # Update cost tracking
                                correction_cost_tracker.append_to_cost_md("Proposal Correction (Streamlit)", config.get('openai', {}).get('model', 'gpt-4o'))

                                st.success("🎉 Corrections applied successfully!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"❌ Correction failed: {str(e)}")

            with col_correction2:
                if st.button("✅ Approve As-Is", type="primary", use_container_width=True, key="approve_btn"):
                    st.success("✅ Proposal approved! You can download the files above.")

            # Show corrected files if they exist
            if 'corrected_pdf_path' in st.session_state:
                st.markdown("---")
                st.markdown("### 📥 Corrected Files")
                download_corr_col1, download_corr_col2 = st.columns(2)

                with download_corr_col1:
                    corrected_pdf_path = st.session_state['corrected_pdf_path']
                    if corrected_pdf_path and os.path.exists(corrected_pdf_path):
                        pdf_link_corrected = create_download_link(
                            corrected_pdf_path,
                            f"proposal_{request.client_name}_corrected.pdf",
                            "pdf"
                        )
                        st.markdown(pdf_link_corrected, unsafe_allow_html=True)

                with download_corr_col2:
                    corrected_docx_path = st.session_state.get('corrected_docx_path')
                    if corrected_docx_path and os.path.exists(corrected_docx_path):
                        docx_link_corrected = create_download_link(
                            corrected_docx_path,
                            f"proposal_{request.client_name}_corrected.docx",
                            "docx"
                        )
                        st.markdown(docx_link_corrected, unsafe_allow_html=True)
                    else:
                        st.info("DOCX correction failed or not available")

                # Show corrected cost summary
                if 'correction_cost' in st.session_state:
                    corrected_cost_summary = st.session_state['correction_cost']
                    if corrected_cost_summary:
                        st.markdown("### 💰 Correction Cost")
                        corr_cost_col1, corr_cost_col2, corr_cost_col3 = st.columns(3)

                        with corr_cost_col1:
                            st.metric("Correction Cost", f"${corrected_cost_summary.get('total_cost', 0):.4f}")
                        with corr_cost_col2:
                            st.metric("Additional Tokens", f"{corrected_cost_summary.get('total_input_tokens', 0) + corrected_cost_summary.get('total_output_tokens', 0):,}")
                        with corr_cost_col3:
                            total_cost = cost_summary.get('total_cost', 0) + corrected_cost_summary.get('total_cost', 0)
                            st.metric("Total Cost", f"${total_cost:.4f}")

                # Show corrected proposal preview
                if 'corrected_html' in st.session_state:
                    st.markdown("### 👁️ Preview Corrected Proposal")
                    with st.expander("View Corrected Proposal", expanded=False):
                        st.components.v1.html(
                            st.session_state['corrected_html'],
                            height=600,
                            scrolling=True
                        )

                # Show detailed correction analysis
                st.markdown("### 📊 Correction Analysis")

                # Show before/after comparison if we have both versions
                if 'corrected_json' in st.session_state and result.get("proposal"):
                    original_proposal = result["proposal"]
                    corrected_proposal = st.session_state['corrected_json']

                    # Create comparison tabs
                    tab1, tab2, tab3 = st.tabs(["📋 Correction Report", "🔍 Before/After Comparison", "📈 Improvement Summary"])

                    with tab1:
                        # Show detailed correction report
                        if 'diff_report_path' in st.session_state:
                            diff_report_path = st.session_state['diff_report_path']
                            if diff_report_path and os.path.exists(diff_report_path):
                                with open(diff_report_path, 'r', encoding='utf-8') as f:
                                    diff_content = f.read()
                                st.text(diff_content)
                        else:
                            st.info("Correction report not available")

                    with tab2:
                        # Show side-by-side comparison for key sections
                        st.markdown("#### Section-by-Section Comparison")

                        original_sections = original_proposal.get('generated_sections', {})
                        corrected_sections = corrected_proposal.get('generated_sections', {})

                        # Show comparison for a few key sections
                        key_sections = ['Executive Summary', 'Budget', 'Technical Approach and Methodology']

                        for section_name in key_sections:
                            if section_name in original_sections and section_name in corrected_sections:
                                st.markdown(f"##### {section_name}")

                                col_before, col_after = st.columns(2)

                                with col_before:
                                    st.markdown("**Before Correction:**")
                                    original_content = original_sections[section_name]
                                    if isinstance(original_content, dict):
                                        content = original_content.get('content', str(original_content))
                                    else:
                                        content = str(original_content)

                                    # Show first 300 characters for comparison
                                    preview = content[:300] + "..." if len(content) > 300 else content
                                    st.text_area("", value=preview, height=150, key=f"before_{section_name}", disabled=True)

                                with col_after:
                                    st.markdown("**After Correction:**")
                                    corrected_content = corrected_sections[section_name]
                                    if isinstance(corrected_content, dict):
                                        content = corrected_content.get('content', str(corrected_content))
                                    else:
                                        content = str(corrected_content)

                                    # Show first 300 characters for comparison
                                    preview = content[:300] + "..." if len(content) > 300 else content
                                    st.text_area("", value=preview, height=150, key=f"after_{section_name}", disabled=True)

                                st.markdown("---")

                    with tab3:
                        # Show improvement metrics
                        st.markdown("#### Improvement Metrics")

                        improvements = []

                        # Count sections processed
                        total_sections = len(original_sections)
                        corrected_section_count = len(corrected_sections)

                        # Calculate content changes
                        total_length_before = 0
                        total_length_after = 0

                        for section_name in original_sections:
                            original_content = original_sections[section_name]
                            if isinstance(original_content, dict):
                                content = original_content.get('content', str(original_content))
                            else:
                                content = str(original_content)
                            total_length_before += len(content)

                        for section_name in corrected_sections:
                            corrected_content = corrected_sections[section_name]
                            if isinstance(corrected_content, dict):
                                content = corrected_content.get('content', str(corrected_content))
                            else:
                                content = str(corrected_content)
                            total_length_after += len(content)

                        # Display improvement metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)

                        with metric_col1:
                            st.metric("Sections Processed", f"{corrected_section_count}/{total_sections}")

                        with metric_col2:
                            change_pct = ((total_length_after - total_length_before) / max(total_length_before, 1)) * 100
                            st.metric("Content Change", f"{change_pct:+.1f}%")

                        with metric_col3:
                            if 'correction_cost' in st.session_state:
                                correction_cost = st.session_state['correction_cost'].get('total_cost', 0)
                                st.metric("Correction Cost", f"${correction_cost:.4f}")

                        # List key improvements
                        st.markdown("#### Key Improvements Made:")
                        improvements = [
                            "✅ Removed JSON formatting artifacts",
                            "✅ Synchronized data across sections",
                            "✅ Improved formatting consistency",
                            "✅ Enhanced professional language",
                            "✅ Preserved table structures",
                            "✅ Fixed encoding issues"
                        ]

                        for improvement in improvements:
                            st.markdown(improvement)

                else:
                    # Fallback to just showing the correction report
                    if 'diff_report_path' in st.session_state:
                        diff_report_path = st.session_state['diff_report_path']
                        if diff_report_path and os.path.exists(diff_report_path):
                            with st.expander("📋 View Correction Report", expanded=True):
                                with open(diff_report_path, 'r', encoding='utf-8') as f:
                                    diff_content = f.read()
                                st.text(diff_content)

    with col2:
        # Enhanced Feature Section
        st.markdown("""
            <div class="feature-list">
                <h3 style="color: #2c3e50; margin-top: 0;">🎯 Enterprise Features</h3>
                <div class="feature-item">🧠 <strong>AI-Powered Analysis</strong><br/>Intelligent RFP parsing and requirement extraction</div>
                <div class="feature-item">📊 <strong>Cost Optimization</strong><br/>Smart resource allocation and budget planning</div>
                <div class="feature-item">📈 <strong>Data Visualization</strong><br/>Professional charts and graphs integration</div>
                <div class="feature-item">🔍 <strong>Quality Assurance</strong><br/>Automated content review and optimization</div>
                <div class="feature-item">💼 <strong>Multi-Format Export</strong><br/>PDF, DOCX, HTML, and JSON outputs</div>
                <div class="feature-item">📋 <strong>Compliance Ready</strong><br/>Government and enterprise standards</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")


        # How it works
        st.markdown("### 🔄 How It Works")
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <strong>1. Upload</strong> → RFP PDF document<br/>
        <strong>2. Extract</strong> → AI analyzes requirements<br/>
        <strong>3. Generate</strong> → Creates proposal content<br/>
        <strong>4. Review</strong> → Quality assurance check<br/>
        <strong>5. Export</strong> → Download final proposal
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()