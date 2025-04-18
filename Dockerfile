# Use the official Python slim image for a smaller footprint
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY app.py .
COPY models/ models/
COPY templates/ templates/
COPY .env .env
# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]