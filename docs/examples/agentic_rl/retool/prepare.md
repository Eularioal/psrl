# ReTool Data & Sandbox Preparation

> **Goal:** prepare the math data and verify that this job’s rollout nodes can reach the configured Python execution service. Preparing data and deploying shared services are separate operations.

Before launching training you need to prepare two things: the **DAPO math parquets**
and the **SandboxFusion code-execution service** that every rollout worker will call
into over HTTP.

---

## Contents

- [1. Data](#1-data)
- [2. SandboxFusion (code-execution service)](#2-sandboxfusion-code-execution-service)
  - [Step 1: Bake the image onto every node](#step-1-bake-the-image-onto-every-node)
  - [Step 2: Deploy the SandboxFusion Swarm service](#step-2-deploy-the-sandboxfusion-swarm-service)
  - [Step 3: Smoke test](#step-3-smoke-test)

## 1. Data

Download the DAPO math parquets into `${PSRL_WORKSPACE}/data/dapo/`:

```
${PSRL_WORKSPACE}/data/dapo/
├── dapo-math-17k.parquet    # training set (BytedTsinghua-SIA/DAPO-Math-17k)
├── aime-2024.parquet        # validation set (optional)
└── aime-2025.parquet        # validation set (default)
```

`CustomRLHFDataset` (in `examples/retool/retool.py`) handles tokenization, caches the
post-mapping dataset under `cache_dir/processed_datasets/<md5>.parquet`, and routes
each row through the right `(problem, answer)` extractor based on the parquet name.

:::{tip}
The download/conversion pattern follows veRL's data-prep conventions, see
**→ [veRL: Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)**
for the underlying parquet schema.
:::

---

## 2. SandboxFusion (code-execution service)

Every rollout worker emits Python `code_interpreter` tool calls that hit
`http://localhost:8080/run_code`. That endpoint is served by a cluster-wide
**SandboxFusion** Swarm service. Bringing it up is a two-step + smoke-test process.

### Step 1: Bake the image onto every node

```bash
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_install.sh

DOCKER_NODE_IPS="${NODE_IPS}" \
DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_copy.sh
```

`docker_install.sh` pulls `code_sandbox:server` via `skopeo` into a shared-FS tar.
`docker_copy.sh` fans the tar out to every node via `pssh` and runs `docker load`.

`PSRL_WORKSPACE` must point to the shared workspace. Set `NODE_IPS` to this job's
allocated node list and replace `DOCKER_NODE_NUM=8` with its actual node count.
`DOCKER_IMAGE_TAG` must name an image accessible from your registry. Set
`DOCKERHUB_MIRROR` only when your deployment explicitly uses a mirror.

### Step 2: Deploy the SandboxFusion Swarm service

This helper changes Swarm membership and replaces its named sandbox service/network.
Use it for a dedicated sandbox deployment whose resources you own. For an existing
shared service, configure its URL in the tool YAML and validate connectivity instead.

```bash
SANDBOX_NODE_IPS="${NODE_IPS}" \
SANDBOX_NODE_NUM=8 \
  bash examples/retool/sandbox_fusion/launch_service.sh
```

This will `docker swarm init` on the first host, `swarm join` everyone else, create
an overlay network, and `docker service create --publish published=8080,target=8080,mode=host`
so every node has a sandbox replica bound to `localhost:8080`.

### Step 3: Smoke test

On any node:

```bash
curl http://localhost:8080/run_code \
    -H 'Content-Type: application/json' \
    --data-raw '{"code": "print(2+3)", "language": "python"}'
```

Expected: `"status": "Success"`, `"stdout": "5\n"`. If this fails from a rollout node,
tool calls cannot receive valid execution results. Inspect service errors separately from task-answer scores.

---

```{seealso}
Full bake / Swarm / troubleshooting docs:
[`examples/retool/docker_scripts/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/retool/docker_scripts/README.md)
and
[`examples/retool/sandbox_fusion/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/retool/sandbox_fusion/README.md).
```
