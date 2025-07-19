# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install ffmpeg and clean up
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir -e .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["deface"]