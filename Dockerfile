# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies including ffmpeg
# Update package list, install ffmpeg, then clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Create the cache directory and set permissions
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 7860

# Define the command to run the application
CMD ["python", "app.py"]