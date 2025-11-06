# Lightweight container to run the decision tree pipeline
FROM python:3.10-slim

# set working dir
WORKDIR /app

# system deps (for some wheels and fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# copy repo files
COPY . /app

# upgrade pip and install python deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Expose notebook port (optional)
EXPOSE 8888

# Default entry: show README and keep container alive for interactive use.
CMD ["bash", "-lc", "echo 'Container ready. Run: python decisiontree_generic.py --data your.csv --output-dir reports --models-dir models' && bash"]