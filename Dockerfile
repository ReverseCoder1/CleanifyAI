FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    openai==1.3.7 \
    pyyaml==6.0.1

# Copy all project files
COPY . .

# Expose port
EXPOSE 7860

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
