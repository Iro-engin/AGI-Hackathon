FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["bash", "-lc", "python src/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace src/kaggle_submission_benchmark.ipynb"]
