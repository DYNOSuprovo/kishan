# Use Python 3.10
FROM python:3.10

# Set working directory to /code
WORKDIR /code

# Copy requirements
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the source code
COPY ./src /code/src

# Create a writable directory for caching/temp files if needed
RUN mkdir -p /code/temp && chmod 777 /code/temp

# Set the command to run the application
# Note: HF Spaces expects the app to run on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]