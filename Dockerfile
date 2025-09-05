# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your app code into the container
COPY . /app

# Install OS-level dependencies (optional but good for LangChain)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
