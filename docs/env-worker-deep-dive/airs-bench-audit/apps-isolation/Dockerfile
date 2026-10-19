FROM python:3.13-slim
RUN pip install --no-cache-dir --timeout 10 --retries 1 datasets==3.6.0
RUN useradd -m -u 1000 agent && mkdir -p /home/agent/workspace /home/agent/miniconda3/envs/mlgym_generic/bin && ln -s /usr/local/bin/python /home/agent/miniconda3/envs/mlgym_generic/bin/python && chown -R agent:agent /home/agent
USER agent
WORKDIR /home/agent/workspace
