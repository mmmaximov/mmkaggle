FROM python:3.11-slim

RUN pip install --no-cache-dir numpy tqdm \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# participant solution: portfolio solver modules
COPY train.py solve.py common.py model.py \
     linear_solver.py slide_solver.py search_v2.py puzzle_solver.py \
     /opt/participant/

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/opt/participant
