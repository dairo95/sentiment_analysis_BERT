# 1. Base Image: Use a lightweight Python runtime
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (optimization for caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the source code
COPY src/ ./src

# 6. Prepare the mount point for the volume
RUN mkdir models
RUN mkdir data

# 7. Expose the API port
EXPOSE 8000

# 8. Entry Point: Launch the API using Uvicorn
CMD ["python3", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]