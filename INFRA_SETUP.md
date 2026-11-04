# End-to-End Google Cloud Infrastructure Setup Guide (GKE + DWS Calendar Mode Reservations for Ironwood TPU v7x)

This guide is designed for **first-time Google Cloud users** who have an active **Dynamic Workload Scheduler (DWS) Calendar Mode reservation** for **Ironwood TPU (`v7x` / `tpu7x-standard-4t`)** chips and want to set up their Google Cloud environment from scratch and run the **`GPT-OSS 120B`** serving and benchmark recipes across **`sglang-torchtpu`**, **`vllm-torchtpu`**, and **`vllm-tpu` (JAX)**.

All benchmarks and recipes in this repository were **tested and validated on Google Kubernetes Engine (GKE)** following Google Cloud's official Ironwood TPU and DWS Calendar Mode reservation architecture:
- [Reserve TPUs in Calendar mode on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode)
- [About TPUs in GKE (Ironwood `tpu7x`)](https://cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood)
- [Serve an LLM using TPU Trillium/Ironwood on GKE with Ray and TorchTPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-llm-tpu-ray-torchtpu)

---

## 1. Architecture FAQ: Why GKE Is Required for Multi-Host Ironwood (`v7x`) & When Standalone GCE Can Be Used

### Q1: How are Ironwood (`v7x`) chips packaged in Google Cloud?
- Each Ironwood TPU host (`tpu7x-standard-4t`) contains **`4` physical TPU v7x chips** (`8` TensorCores) connected over an ultra-fast **`2x2x1` Inter-Chip Interconnect (ICI)** mesh, with **`192 GB` of HBM3e per chip** (**`768 GB` total HBM3e per host**) and `192` vCPUs / `1,536 GB` system RAM.
- Depending on your DWS Calendar reservation size, your workload runs on:
  - **1 Host (`4` physical chips, `2x2x1` topology)** = `1 × tpu7x-standard-4t` (`768 GB` HBM3e)
  - **2 Hosts (`8` physical chips, `2x2x2` topology)** = `2 × tpu7x-standard-4t` (`1.54 TB` HBM3e)
  - **4 Hosts (`16` physical chips, `2x2x4` topology)** = `4 × tpu7x-standard-4t` (`3.07 TB` HBM3e)

### Q2: Do we need GKE, or can we run directly on raw GCE VMs?
- **For Multi-Host (`2` hosts = `8` chips `2x2x2`, or `4` hosts = `16` chips `2x2x4`) and Production Workloads — GKE is Mandatory / Strongly Recommended**:
  1. **Official Placement & Workload Policies (`HIGH_THROUGHPUT`)**: On Ironwood (`tpu7x`), legacy `gcloud compute tpus tpu-vm` APIs are superseded by Compute Engine `tpu7x-standard-4t` instances paired with regional **Workload Policies (`--type=HIGH_THROUGHPUT --accelerator-topology=2x2x2` or `2x2x4`)**. GKE node pools natively bind your DWS Calendar Mode reservation (`--reservation-affinity=specific --reservation=<RESERVATION_NAME>`) and your Workload Policy (`--placement-policy=<POLICY_NAME>`) so all hosts in a `2x2x2` or `2x2x4` slice are co-scheduled atomically on the optical ICI fabric.
  2. **Automated Multi-Host TorchTPU / JAX Coordination**: GKE's TPU webhook, KubeRay (`RayOperator`), and LeaderWorkerSet (LWS) automatically inject required cross-host TPU runtime environment variables (`TPU_WORKER_ID`, `TPU_WORKER_HOSTNAMES`, `TORCH_TPU_TOPOLOGY`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`), manage `/dev/vfio/*` device permissions, and restart all workers cleanly if a node fails.
  3. **Built-in Load Balancing & Shared Storage**: GKE provides native `GcsFuseCsiDriver` volume mounts for `234 GB` BF16 / `65.3 GB` MXFP4 weights and shared XLA compilation caches, plus native Kubernetes `Service` / Ray Serve LLM / GKE Inference Gateway load balancing across multi-host replicas without requiring manual external NGINX VMs.
- **For Single-Host (`1` host = `4` chips `2x2x1`) Basic Testing Only**:
  - Even for a single host, **GKE is our primary tested and recommended path** because you use the exact same Kubernetes manifests (`k8s/*.yaml`) for 1, 2, or 4 hosts.
  - However, if you only want to test **1 host (`4` chips)** on a standalone VM without creating a GKE cluster, you can provision a single `tpu7x-standard-4t` GCE VM targeting your DWS Calendar reservation (see [Appendix A](#appendix-a-optional-single-host-1-host--4-chips-standalone-gce-vm-testing)).

---

## 2. Understanding `GPT-OSS 120B` Precision Formats & TorchTPU Runtime Patches on Ironwood TPU (`v7x`)

Before launching your benchmarks, it is helpful to understand the two official checkpoints of **`GPT-OSS 120B`** and how they execute on **`sglang-torchtpu`**, **`vllm-torchtpu`**, and **`vllm-tpu` (JAX)**:

| Checkpoint | Precision & Weight Size | Supported Serving Stacks on TPU `v7x` | Recommended PoC & Production Use Case |
|---|---|---|---|
| **[`lmsys/gpt-oss-120b-bf16`](https://huggingface.co/lmsys/gpt-oss-120b-bf16)** | **Pure Unquantized BF16** (`233.7 GB` weights) | **`sglang-torchtpu`**, **`vllm-torchtpu`**, AND **`vllm-tpu` (JAX)** | **Best for Apples-to-Apples TorchTPU (`sglang-torchtpu` & `vllm-torchtpu`) vs. `vllm-tpu` Evaluation** and zero-quantization accuracy |
| **[`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)** | **Native MXFP4 MoE + BF16 Attention** (`65.3 GB` weights) | **`sglang-torchtpu`** (`TP=8, EP=8`) AND **`vllm-tpu` (JAX)** (`TP=2, DP=4`) | **Best for Maximum Throughput (`vllm-tpu` `TP=2, DP=4`)** or **Ultra-Low Latency (`sglang-torchtpu` `4.09 ms` TPOT)** |

### FAQ 1: Why does `openai/gpt-oss-120b` (Native MXFP4) work on `sglang-torchtpu` and `vllm-tpu`, but not on `vllm-torchtpu`?
- OpenAI released `openai/gpt-oss-120b` using **`gpt_oss_mxfp4`**, a 4-bit microscaling format (`32`-element blocks with `E8M0` shared exponents) originally designed around GPU/CUDA Triton kernels.
- Ironwood TPU (`v7x`) Matrix Multiply Units (MXUs) execute natively in **BF16 (`2,307 TFLOPs/chip`)** and **FP8 (`4,614 TFLOPs/chip`)** (native hardware FP4 MXU arrives in TPU v8).
- **`sglang-torchtpu`** and **`vllm-tpu` (JAX)** include custom Pallas TPU kernels that unpack the 4-bit `mxfp4` weights on the fly inside the TPU Vector Processing Unit (VPU) into BF16/FP8 registers right before feeding the MXU. That is why `sglang-torchtpu` serves `openai/gpt-oss-120b` natively at **`737.0 ms` TTFT** and **`4.09 ms` TPOT**, and `vllm-tpu` serves 4 data-parallel replicas per host (`TP=2, DP=4`) at **`5,847.0` out tok/s** (`1` host) and **`20,366.9` out tok/s** (`4` hosts).
- By contrast, upstream **`vllm-torchtpu`** has not yet registered a TPU backend kernel for `gpt_oss_mxfp4` (`vllm/model_executor/layers/quantization/mxfp4.py` only binds CUDA/ROCm Triton kernels) and raises `ValueError: gpt_oss_mxfp4 quantization is currently not supported in tpu`.

### FAQ 2: Why did `lmsys/gpt-oss-120b-bf16` (Native Unquantized BF16) need a lightweight runtime patch on `vllm-torchtpu` and `sglang-torchtpu`?
When running the pure unquantized BF16 checkpoint (`lmsys/gpt-oss-120b-bf16`, `234 GB` weights — which easily fits inside a single 4-chip Ironwood host's `768 GB` HBM3e at `TP=8, EP=8`), **`vllm-tpu` (JAX)** works out-of-the-box with zero code changes (`2,540.3` out tok/s on `1k/1k`).

For the **TorchTPU ecosystem (`vllm-torchtpu` and `sglang-torchtpu`)**, we provide two small, automated load-time scripts in [`scripts/patches/`](scripts/patches/README.md) (`apply_vllm_torchtpu_bf16_patches.py` and `apply_sglang_bf16_patches.py`) that solve two upstream model-loader details at container startup:
1. **Standard BF16 Checkpoint Tensor Mapping**: Because OpenAI originally released `gpt-oss-120b` only in MXFP4, upstream `vllm-torchtpu` and `sglang-torchtpu` wrote their `gpt_oss.py` weight loaders assuming MXFP4 block/scale tensor names (`gate_up_proj_blocks`). Our startup script maps standard Hugging Face BF16 expert tensor names (`experts.gate_up_proj`, `experts.down_proj`, and biases) and deinterleaves OpenAI's interleaved `swigluoai` gate/up channels (`[::2]` and `[1::2]`) once at load time.
2. **TPU MXU `128 × 128` Tile Alignment (`hidden_size = 2880 -> 2944`)**:
   - TPU Matrix Multiply Units (MXUs) and Pallas grouped-matmul (`gmm_v2`) kernels operate on **`128 × 128` hardware tiles**.
   - Most LLMs use a `hidden_size` that is an exact multiple of `128` (e.g., `2048`, `4096`, `5120`). However, **`GPT-OSS 120B` has `hidden_size = 2880`**, which is `22.5 × 128` (leaving an unaligned remainder of `64` columns).
   - In `vllm-torchtpu`'s unquantized BF16 MoE layer (`VllmUnquantizedFusedMoEMethod`), upstream padded `intermediate_size` to a multiple of `128`, **but forgot to pad `hidden_size` (`2880 -> 2944`, adding `64` zero columns)**. Without that 6-line load-time padding, Pallas `gmm_v2` reads 64 uninitialized memory lanes past each row (`18.8 ms` TPOT).
   - With `python3 scripts/patches/apply_vllm_torchtpu_bf16_patches.py` applied before server launch, **`vllm-torchtpu` runs Native BF16 `GPT-OSS 120B` cleanly at `5.30–5.95 ms` TPOT (`3.2x` faster decode) and `2,284.8` out tok/s on `1k/1k`**!
   - With `python3 scripts/patches/apply_sglang_bf16_patches.py` applied before server launch, **`sglang-torchtpu` achieves the #1 lowest latency across all stacks on Native BF16: `739.5 ms` TTFT on `22k` input, `4.97 ms` TPOT (`201 tok/s` per stream), `2,530.8` out tok/s on 1 host (`4` chips), and `5,084.5` out tok/s (`2.01x` linear scaling) on 2 hosts (`8` chips)**!

---

## 3. Prerequisites & Variables (Set Once in Your Terminal)

Install the [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) and `kubectl` (`gcloud components install kubectl gke-gcloud-auth-plugin`), or open **Google Cloud Shell** in the Console. Then set your environment variables:

```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="us-central1"                     # Region where your DWS Calendar reservation lives
export ZONE="us-central1-c"                     # Specific zone of your DWS Calendar reservation
export RESERVATION_NAME="<YOUR_DWS_CALENDAR_RESERVATION_NAME>"

export CLUSTER_NAME="tpu-v7x-bench-cluster"
export VPC_NAME="tpu-v7x-vpc"
export SUBNET_NAME="tpu-v7x-subnet"
export GCS_BUCKET="${PROJECT_ID}-tpu-v7x-models"
export HF_TOKEN="<YOUR_HUGGINGFACE_READ_TOKEN>" # Must have access to openai/gpt-oss-120b & lmsys/gpt-oss-120b-bf16

gcloud auth login
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"
```

> **One-Command Automated GKE + DWS Calendar Provisioning**:
> You can run [`k8s/provision_gke_dws_calendar.sh`](k8s/provision_gke_dws_calendar.sh) (`NUM_HOSTS=1`, `2`, or `4`) to execute Steps 4 through 7 automatically, or follow the step-by-step commands below.

---

## 4. Enable Required Google Cloud APIs, VPC, Cloud NAT & GCS Bucket

```bash
# 1. Enable Compute, GKE, TPU, Storage, and Artifact Registry APIs
gcloud services enable \
  compute.googleapis.com \
  container.googleapis.com \
  tpu.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com

# 2. Create custom VPC & Subnet with Private Google Access (for high-speed GCS weight streaming)
gcloud compute networks create "${VPC_NAME}" --subnet-mode=custom

gcloud compute networks subnets create "${SUBNET_NAME}" \
  --network="${VPC_NAME}" \
  --region="${REGION}" \
  --range="10.128.0.0/16" \
  --enable-private-ip-google-access

# 3. Create Cloud Router & Cloud NAT (allows private GKE nodes to pull container images & Hugging Face weights)
gcloud compute routers create "${VPC_NAME}-router" \
  --network="${VPC_NAME}" \
  --region="${REGION}"

gcloud compute routers nats create "${VPC_NAME}-nat" \
  --router="${VPC_NAME}-router" \
  --region="${REGION}" \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges

# 4. Create regional GCS bucket for model weights & persistent XLA compilation cache
gcloud storage buckets create "gs://${GCS_BUCKET}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

---

## 5. Verify Your Active DWS Calendar Mode Reservation

When your DWS Calendar reservation window starts, Compute Engine creates an active specific reservation in your zone. Verify its exact name, machine type (`tpu7x-standard-4t`), and available host count (`count` vs `inUseCount`):

```bash
gcloud compute reservations list \
  --filter="zone:(${ZONE})" \
  --format="table(name,status,zone,specificReservation.count,specificReservation.inUseCount,specificReservation.instanceProperties.machineType)"

# Set RESERVATION_NAME to the exact name shown above:
export RESERVATION_NAME="<YOUR_DWS_CALENDAR_RESERVATION_NAME>"
```

- `count: 1` -> **1 Host (`4` physical `v7x` chips, `2x2x1`)**
- `count: 2` -> **2 Hosts (`8` physical `v7x` chips, `2x2x2`)**
- `count: 4` -> **4 Hosts (`16` physical `v7x` chips, `2x2x4`)**

---

## 6. Create the Standard GKE Cluster & Bind Your DWS Calendar TPU v7x Node Pool

### Step 6.1: Create the GKE Cluster (with GCSFuse CSI Driver & KubeRay Operator Enabled)

```bash
gcloud container clusters create "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --network="${VPC_NAME}" \
  --subnetwork="${SUBNET_NAME}" \
  --release-channel=rapid \
  --machine-type=e2-standard-8 \
  --num-nodes=2 \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --addons=GcsFuseCsiDriver,RayOperator \
  --scopes=https://www.googleapis.com/auth/cloud-platform

gcloud container clusters get-credentials "${CLUSTER_NAME}" --region="${REGION}"

# Grant node service account access to your GCS model bucket
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### Step 6.2: Create the Ironwood TPU (`tpu7x-standard-4t`) Node Pool Consuming Your DWS Calendar Reservation

Choose **Option A (1 Host = 4 Chips)**, **Option B (2 Hosts = 8 Chips)**, or **Option C (4 Hosts = 16 Chips)** matching your reservation size:

#### Option A: Single-Host (`1` Host = `4` Physical `v7x` Chips, `2x2x1` Topology)
```bash
gcloud container node-pools create tpu-v7x-1host-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=1 \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

#### Option B: 2-Host (`2` Hosts = `8` Physical `v7x` Chips, `2x2x2` Topology)
Per Google Cloud's [Reserve TPUs in Calendar mode on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/tpu-calendar-mode) guide, multi-host Ironwood node pools require a regional **`HIGH_THROUGHPUT` Workload Policy** specifying the `2x2x2` accelerator topology:
```bash
gcloud compute resource-policies create workload-policy tpu7x-2x2x2-policy \
  --region="${REGION}" \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x2

gcloud container node-pools create tpu-v7x-2host-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=2 \
  --placement-policy=tpu7x-2x2x2-policy \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

#### Option C: 4-Host (`4` Hosts = `16` Physical `v7x` Chips, `2x2x4` Topology)
```bash
gcloud compute resource-policies create workload-policy tpu7x-2x2x4-policy \
  --region="${REGION}" \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=2x2x4

gcloud container node-pools create tpu-v7x-4host-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=4 \
  --placement-policy=tpu7x-2x2x4-policy \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Step 6.3: Create a Dedicated CPU Benchmark Client Node Pool (`c2-standard-16`)
Running `vllm bench serve` on a dedicated CPU node pool prevents client tokenization and async HTTP load generation from stealing host CPU cores from the TPU runtime:
```bash
gcloud container node-pools create cpu-bench-pool \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --node-locations="${ZONE}" \
  --machine-type=c2-standard-16 \
  --num-nodes=1 \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### Step 6.4: Verify TPU Node Pool Readiness
```bash
kubectl get nodes -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology
```
Each `tpu7x-standard-4t` node will report `google.com/tpu: 4` (`4` physical Ironwood chips = `8` TensorCores).

---

## 7. Stage `GPT-OSS 120B` Checkpoints to Your GCS Bucket (Run Once)

Stage both the **Pure Unquantized BF16** (`lmsys/gpt-oss-120b-bf16`, `233.7 GB`) and **Native MXFP4** (`openai/gpt-oss-120b`, `65.3 GB`) checkpoints into your regional GCS bucket using [`scripts/stage_hf_to_gcs.py`](scripts/stage_hf_to_gcs.py):

```bash
pip install -U huggingface_hub google-cloud-storage

# 1. Stage Pure Unquantized BF16 checkpoint (lmsys/gpt-oss-120b-bf16, ~234 GB)
python3 scripts/stage_hf_to_gcs.py \
  --repo-id lmsys/gpt-oss-120b-bf16 \
  --gcs-bucket "${GCS_BUCKET}" \
  --gcs-prefix models/lmsys/gpt-oss-120b-bf16

# 2. Stage Native MXFP4 checkpoint (openai/gpt-oss-120b, ~65.3 GB)
python3 scripts/stage_hf_to_gcs.py \
  --repo-id openai/gpt-oss-120b \
  --gcs-bucket "${GCS_BUCKET}" \
  --gcs-prefix models/openai/gpt-oss-120b
```

---

## 8. Deploy & Benchmark `GPT-OSS 120B` on GKE (Native BF16 & MXFP4 Across `sglang-torchtpu`, `vllm-torchtpu`, and `vllm-tpu`)

### 8.1 Deploy Serving Pods on GKE (`1`, `2`, or `4` Hosts)

All Kubernetes manifests in [`k8s/`](k8s/) mount your GCS bucket via `gke-gcsfuse-volumes`, configure `/dev/shm`, and automatically apply the startup patches (`scripts/patches/`) when needed. Simply substitute `<YOUR_GCS_BUCKET>` with `${GCS_BUCKET}`:

#### Option 1: Pure Unquantized BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB`) on `sglang-torchtpu` (`TP=8, EP=8` per host — Lowest Latency: `739.5 ms` TTFT, `4.97 ms` TPOT)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-bf16-sglang-v7x.yaml | kubectl apply -f -

# For 2 hosts (8 chips -> 5,084.5 out tok/s = 2.01x scaling) or 4 hosts (16 chips), scale replicas:
kubectl scale deployment/gpt-oss-120b-bf16-sglang-v7x --replicas=2
```

#### Option 2: Pure Unquantized BF16 (`lmsys/gpt-oss-120b-bf16`, `234 GB`) on `vllm-tpu` (JAX `TP=8, --enable-expert-parallel` per host — Zero Patches: `2,540.3` out tok/s)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-bf16-vllm-tpu-v7x.yaml | kubectl apply -f -

# Scale to 2 hosts (8 chips) or 4 hosts (16 chips):
kubectl scale deployment/gpt-oss-120b-bf16-vllm-tpu-v7x --replicas=2
```

#### Option 3: Native MXFP4 (`openai/gpt-oss-120b`, `65.3 GB`) on `vllm-tpu` (`TP=2, DP=4` per host — Peak Throughput: `5,847` tok/s on 1 host, `14,277` on 2 hosts, `20,367` on 4 hosts)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-vllm-tpu-v7x.yaml | kubectl apply -f -

# Scale to 2 hosts (8 chips) or 4 hosts (16 chips):
kubectl scale deployment/gpt-oss-120b-vllm-tpu-v7x --replicas=2
```

#### Option 4: Native MXFP4 (`openai/gpt-oss-120b`, `65.3 GB`) on `sglang-torchtpu` (`TP=8, EP=8` + Optional `EAGLE3` Speculative Decoding — `4.09 ms` TPOT)
```bash
sed "s/<YOUR_GCS_BUCKET>/${GCS_BUCKET}/g" k8s/gpt-oss-120b-sglang-eagle3-v7x.yaml | kubectl apply -f -
```

### 8.2 Expose a Kubernetes `ClusterIP` Service & Run Benchmarks from the CPU Node Pool

```bash
# Expose your active deployment as a ClusterIP Service on port 8000
kubectl expose deployment gpt-oss-120b-bf16-sglang-v7x \
  --name=gpt-oss-120b-svc \
  --port=8000 \
  --target-port=8000 \
  --type=ClusterIP

# Watch server logs until warmup & XLA compilation finish
kubectl logs -f deploy/gpt-oss-120b-bf16-sglang-v7x
```

Once `/v1/models` returns HTTP 200, run the full reproducible benchmark suite (`configs/bf16/*.json` or `configs/*.json`) or `vllm bench serve` from a pod scheduled on `cpu-bench-pool`:

```bash
kubectl run bench-client --rm -it --restart=Never \
  --image=vllm/vllm-tpu:latest \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"cpu-bench-pool"}}}' \
  -- bash -c "
    vllm bench serve \
      --backend vllm \
      --base-url http://gpt-oss-120b-svc.default.svc.cluster.local:8000 \
      --model openai/gpt-oss-120b \
      --tokenizer lmsys/gpt-oss-120b-bf16 \
      --dataset-name random \
      --random-input-len 1024 \
      --random-output-len 1024 \
      --max-concurrency 64 \
      --num-prompts 256 \
      --ignore-eos \
      --trust-remote-code
  "
```

> **Tip for Multi-Host Long-Context (`22k / 4k`) Benchmarking**:
> When scaling across 2 or 4 hosts (`--replicas=2` or `4`), you can also run [`scripts/lb_proxy.py`](scripts/lb_proxy.py) (`python3 scripts/lb_proxy.py --port 8000 --backends http://<POD_0_IP>:8000 http://<POD_1_IP>:8000`) inside the benchmark pod for active least-connections (`least_conn`) routing across long-context (`22k/4k`) requests.

---

## Appendix A: Optional Single-Host (`1` Host = `4` Chips) Standalone GCE VM Testing

If you only want to test a **single host (`1 × tpu7x-standard-4t` = `4` chips)** without creating a GKE cluster, you can create a standalone GCE VM consuming 1 host from your DWS Calendar reservation:

```bash
gcloud compute instances create tpu-v7x-single-host \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type=tpu7x-standard-4t \
  --network-interface="network=${VPC_NAME},subnet=${SUBNET_NAME},no-address" \
  --reservation-affinity=specific \
  --reservation="${RESERVATION_NAME}" \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=hyperdisk-balanced \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Then SSH into `tpu-v7x-single-host` (`gcloud compute ssh tpu-v7x-single-host --zone="${ZONE}" --tunnel-through-iap`) and use the single-host helper script [`gce/launch_gce_tpu_v7x_gpt_oss_120b.sh`](gce/launch_gce_tpu_v7x_gpt_oss_120b.sh) (`serve-bf16-sglang`, `serve-bf16-torchtpu`, `serve-bf16-vllm-tpu`, `serve-mxfp4-vllm-tpu`, or `serve-mxfp4-sglang`). For any **2-host (`8`-chip) or 4-host (`16`-chip)** deployment, always use the **GKE workflow in Sections 6–8**.
