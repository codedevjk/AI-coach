# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the dependencies
# Update pip first, then install requirements
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create the cache directory
# Ensure the 'app' user (or default user, often root in this base image) owns it
# Using 'chmod 777' is a broad fix, but ensures read/write/execute for all.
# A more secure approach would be to create a specific user, but this is simpler for now.
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
# This should match the app_port in README.md
EXPOSE 7860

# Define the command to run the application
# Gradio apps typically run on 0.0.0.0 to be accessible from outside the container
CMD ["python", "app.py"]