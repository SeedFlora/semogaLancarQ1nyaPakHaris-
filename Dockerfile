# Reproducible environment for the per-sample calibration / paired-test /
# selective-prediction analysis of the SmartCity IJOST manuscript.
#
#   docker build -t q1-uncertainty .
#   docker run --rm q1-uncertainty                      # run the test suite (synthetic data)
#   docker run --rm -v "${PWD}:/work" q1-uncertainty \
#       python analyze_uncertainty.py --preds /work/per_sample_predictions --out /work/results
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime used by CatBoost's CPU inference
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "pytest", "-q", "tests"]
