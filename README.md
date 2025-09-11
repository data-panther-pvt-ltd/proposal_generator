# AzmX Proposal Generator

An enterprise-grade proposal generation system using OpenAI's native APIs with multi-agent orchestration.

## Tech Stack

- **AI/LLM**: OpenAI GPT-4, LangChain, OpenAI Agents
- **Backend**: Python 3.9+, FastAPI/AsyncIO
- **Frontend**: Streamlit with authentication
- **PDF Generation**: WeasyPrint, PyPDF2, pdfplumber
- **Data Visualization**: Plotly, Kaleido, Pandas, NumPy
- **Vector Database**: FAISS for embeddings
- **Web Search**: Tavily API integration
- **Document Processing**: Jinja2 templates, Markdown, BeautifulSoup4, python-docx, pypandoc
- **Development Tools**: pytest, black, flake8
- **Containerization**: Docker

## Project Summary

The AzmX Proposal Generator is an intelligent document generation system that leverages multiple AI agents to create comprehensive business proposals. The system employs six specialized agents that work collaboratively: a Research Agent for market analysis, a Technical Agent for solution architecture, a Budget Agent for financial planning, a Timeline Agent for project scheduling, a Risk Agent for risk assessment, and a Quality Agent for content validation. 

The application features a user-friendly Streamlit interface with authentication, accepts RFP documents and company profiles as inputs, and generates professional proposals complete with Gantt charts, budget breakdowns, and risk matrices. It integrates web search capabilities for real-time market insights and exports high-quality PDF documents with modern design templates. The system is built with enterprise-grade security, including API key management and session timeouts, making it suitable for production deployment in professional environments.

## Features

- 🤖 **Multi-Agent System**: 6 specialized agents working together
- 📊 **Professional Charts**: Gantt charts, budget breakdowns, risk matrices
- 📄 **PDF Export**: High-quality PDF generation with WeasyPrint
- 🔍 **Web Research**: Integrated Tavily search for market insights
- ✅ **Quality Assurance**: Automated content evaluation and scoring
- 🎨 **Beautiful Output**: Professional HTML templates with modern design

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Tavily API key (optional, for web search)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/azmx/proposal-generator.git
cd proposal-generator