# Start from the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Install Python and other necessary tools
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the timezone to Shanghai
RUN ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get update && apt-get install -y tzdata

# Set the environment variable for timezone
ENV TZ=Asia/Shanghai

# Install additional libraries required for certain Python packages (e.g., OpenCV)
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set the working directory to /app inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install the Python dependencies specified in requirements.txt
RUN pip3 install -r requirements.txt

# Run a Python script for model initialization
RUN python3 model_initialization.py

# Expose port 5000 for external access (e.g., for a Flask app)
EXPOSE 5000

# Define the command to run when the container starts (here, it runs a Python script named app.py)
CMD ["python3", "app.py"]
