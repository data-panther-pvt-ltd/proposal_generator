# Use Python 3.11 as the base image
FROM python:3.11-slim

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 \
    libasound2 libatspi2.0-0 libgtk-3-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Setup crawl4ai and playwright
RUN playwright install chromium


# Expose port
EXPOSE 8501 8000

# Run streamlit
CMD bash -c "uvicorn app.website_evaluation_main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_main.py --server.address=0.0.0.0"
